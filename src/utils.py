import os
import cv2
import numpy as np

from config import *


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return image


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_image(path, image):
    cv2.imwrite(path, image)


# ---------------------------------------------------------------
# STAGE 1: Solder segmentation
# ---------------------------------------------------------------

def segment_solders(gray):
    """
    Segment solder (pad/lead) regions from background and thin traces.
    Solders are the solid mid-gray blobs (leads, big square pad, etc).
    """
    blurred = cv2.GaussianBlur(gray, SOLDER_BLUR_KERNEL, 0)

    # Otsu split: background/thin traces vs solid solder bodies
    _, binary = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SOLDER_MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    solders = []
    for c in contours:
        area = cv2.contourArea(c)
        if SOLDER_MIN_AREA <= area <= SOLDER_MAX_AREA:
            solders.append(c)

    return solders, binary


def classify_solder(contour):
    """
    Bucket a solder contour into a shape class based on its area and
    aspect ratio (long side / short side of the min-area rectangle).

    Returns one of: "solder_long", "solder_large", "solder_middle", "solder"
    """
    area = cv2.contourArea(contour)
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    long_side, short_side = max(w, h), min(w, h)
    aspect_ratio = (long_side / short_side) if short_side > 0 else 0

    if aspect_ratio >= SOLDER_LONG_ASPECT_THRESH:
        return "solder_long"
    if area >= SOLDER_LARGE_AREA_THRESH:
        return "solder_large"
    if area >= SOLDER_MIDDLE_AREA_THRESH:
        return "solder_middle"
    return "solder"


# ---------------------------------------------------------------
# STAGE 2: Void detection, local to each solder
# ---------------------------------------------------------------

def stretch_contrast(roi_gray, roi_mask):
    """
    Local CLAHE + min-max stretch confined to masked pixels only,
    so dim/low-contrast solders get their voids pulled out too.
    """
    masked_vals = roi_gray[roi_mask == 255]
    if masked_vals.size == 0:
        return roi_gray

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(roi_gray)

    # Min-max stretch using only in-mask pixel range
    vmin = np.percentile(masked_vals, 2)
    vmax = np.percentile(masked_vals, 98)
    if vmax <= vmin:
        return enhanced

    stretched = np.clip((enhanced.astype(np.float32) - vmin) * 255.0 / (vmax - vmin), 0, 255)
    return stretched.astype(np.uint8)


def find_voids_in_solder(gray, solder_contour, void_params=None):
    """
    Detect voids strictly inside one solder's mask.
    Voids are brighter blobs than surrounding solder -> local Otsu
    on contrast-stretched ROI, restricted to the mask.

    void_params: optional dict with keys min_area, max_area,
    min_circularity, min_solidity. Falls back to the global VOID_*
    config defaults when not provided (or when a key is missing) so
    this stays backward compatible with old call sites.
    """
    params = void_params or {}
    min_area = params.get("min_area", VOID_MIN_AREA)
    max_area = params.get("max_area", VOID_MAX_AREA)
    min_circularity = params.get("min_circularity", VOID_MIN_CIRCULARITY)
    min_solidity = params.get("min_solidity", VOID_MIN_SOLIDITY)
    x, y, w, h = cv2.boundingRect(solder_contour)
    pad = 2
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])

    roi_gray = gray[y0:y1, x0:x1]
    if roi_gray.size == 0:
        return []

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [solder_contour], -1, 255, -1)
    roi_mask = mask[y0:y1, x0:x1]

    if np.count_nonzero(roi_mask) < min_area:
        return []

    enhanced = stretch_contrast(roi_gray, roi_mask)

    # Erode mask slightly so solder edge/background bleed isn't picked as void
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_mask = cv2.erode(roi_mask, erode_kernel, iterations=1)

    _, local_binary = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    local_binary = cv2.bitwise_and(local_binary, local_binary, mask=inner_mask)

    # Clean speckle noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, clean_kernel)

    contours, _ = cv2.findContours(
        local_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    voids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        c_shifted = c + [x0, y0]
        voids.append(c_shifted)

    return voids


# ---------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------

def draw_results(image, classified_solders, voids):
    """
    classified_solders: list of (contour, class_name) tuples. The class
    is not used for coloring here (every solder is drawn the same
    color) -- it's only kept around because generate_masks.py still
    needs it to pick per-class void params before calling this.
    voids: flat list of void contours (already in full-image coords),
    covering both big and small voids across all solders.
    """
    output = image.copy()

    solder_contours = [contour for contour, _ in classified_solders]
    cv2.drawContours(output, solder_contours, -1, SOLDER_COLOR, 2)
    cv2.drawContours(output, voids, -1, VOID_COLOR, 2)
    return output


def build_void_mask(image_shape, voids):
    """
    Build a single-channel binary mask (0/255), same H x W as the source
    image, with every void filled in white. This is the actual "void
    mask" -- not a visualization, just the raw detected regions.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if voids:
        cv2.drawContours(mask, voids, -1, 255, thickness=-1)
    return mask