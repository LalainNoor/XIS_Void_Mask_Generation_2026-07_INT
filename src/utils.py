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
    blurred = cv2.GaussianBlur(gray, SOLDER_BLUR_KERNEL, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, SOLDER_MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = gray.shape[:2]
    solders = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (SOLDER_MIN_AREA <= area <= SOLDER_MAX_AREA):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        touches_border = (x <= BORDER_MARGIN or y <= BORDER_MARGIN or
                          x + cw >= img_w - BORDER_MARGIN or
                          y + ch >= img_h - BORDER_MARGIN)
        if touches_border and area < SOLDER_MIN_AREA_BORDER_TOUCH:
            continue
        if touches_border:
            perimeter = cv2.arcLength(c, True)
            border_circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            if border_circ < SOLDER_MIN_CIRCULARITY_BORDER:
                continue
            tmp2 = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(tmp2, [c], -1, 255, -1)
            if gray[tmp2 == 255].mean() > 115:
                continue
            _, _, cw2, ch2 = cv2.boundingRect(c)
            aspect = max(cw2, ch2) / (min(cw2, ch2) + 1e-5)
            if aspect > SOLDER_MAX_ASPECT_BORDER:
                continue
        spans_full_extent = (ch >= BORDER_SPAN_FRACTION * img_h or cw >= BORDER_SPAN_FRACTION * img_w)
        if touches_border and spans_full_extent:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < SOLDER_MIN_SOLIDITY:
            continue
        tmp = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(tmp, [c], -1, 255, -1)
        roi_pixels = gray[tmp == 255]
        mean_val = roi_pixels.mean()
        bright_ratio = np.sum(roi_pixels > 200) / roi_pixels.size
        if mean_val > SOLDER_MAX_MEAN_BRIGHTNESS or bright_ratio > SOLDER_BRIGHT_PIXEL_RATIO:
            continue
        solders.append(c)
    return solders, binary


def classify_solder(contour):
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
# STAGE 2: Void detection
# ---------------------------------------------------------------

def stretch_contrast(roi_gray, roi_mask):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    enhanced = clahe.apply(roi_gray)
    masked_vals = enhanced[roi_mask == 255]
    if masked_vals.size == 0:
        return enhanced
    vmin = np.percentile(masked_vals, 2)
    vmax = np.percentile(masked_vals, 99)
    if vmax <= vmin:
        return enhanced
    stretched = np.clip((enhanced.astype(np.float32) - vmin) * 255.0 / (vmax - vmin), 0, 255)
    return stretched.astype(np.uint8)


def _tophat_kernel_for(max_area):
    diameter = int(2 * np.sqrt(max_area / np.pi))
    size = max(9, diameter + 6)
    if size % 2 == 0:
        size += 1
    return (size, size)


def find_voids_in_solder(gray, solder_contour, void_params=None):
    params = void_params or {}
    min_area         = params.get("min_area",            VOID_MIN_AREA)
    max_area         = params.get("max_area",            VOID_MAX_AREA)
    min_circularity  = params.get("min_circularity",     VOID_MIN_CIRCULARITY)
    min_solidity     = params.get("min_solidity",        VOID_MIN_SOLIDITY)
    max_aspect       = params.get("max_aspect_ratio",    VOID_MAX_ASPECT_RATIO)
    min_brt_ratio    = params.get("min_brightness_ratio",VOID_MIN_BRIGHTNESS_RATIO)
    use_tophat       = params.get("use_tophat", True)

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

    solder_area = np.count_nonzero(roi_mask)
    if solder_area < min_area:
        return []

    # Solder mean brightness — used later for brightness-ratio filter
    solder_mean = float(roi_gray[roi_mask == 255].mean())

    enhanced = stretch_contrast(roi_gray, roi_mask)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive erosion: larger solders get more edge-strip removed
    erode_iters = 2 if solder_area > 5000 else 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_mask = cv2.erode(roi_mask, erode_kernel, iterations=erode_iters)

    # Pass 1: global Otsu / Triangle on CLAHE-stretched ROI
    threshold_mode = (cv2.THRESH_BINARY + cv2.THRESH_OTSU if max_area > 2000
                      else cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    _, otsu_binary = cv2.threshold(enhanced, 0, 255, threshold_mode)

    # Pass 2: white top-hat (disabled for large/long pads to avoid gradient FP)
    if use_tophat:
        tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _tophat_kernel_for(max_area))
        tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, tophat_kernel)
        masked_tophat = tophat[inner_mask == 255].astype(np.float32)
        if masked_tophat.size > 0 and masked_tophat.max() > 0:
            vmax = np.percentile(masked_tophat, 99)
            if vmax > 0:
                tophat_stretched = np.clip(tophat.astype(np.float32) * 255.0 / vmax, 0, 255).astype(np.uint8)
                _, tophat_binary = cv2.threshold(tophat_stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                tophat_binary = np.zeros_like(roi_gray)
        else:
            tophat_binary = np.zeros_like(roi_gray)
        local_binary = cv2.bitwise_or(otsu_binary, tophat_binary)
    else:
        local_binary = otsu_binary
    local_binary = cv2.bitwise_and(local_binary, local_binary, mask=inner_mask)

    # Clean speckle — slightly larger kernel than before to kill single-pixel noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, clean_kernel)

    contours, _ = cv2.findContours(local_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    voids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        # Circularity
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        # NEW: aspect ratio — reject wire-bond shadows & edge streaks
        vx, vy, vw, vh = cv2.boundingRect(c)
        aspect = max(vw, vh) / (min(vw, vh) + 1e-5)
        if aspect > max_aspect:
            continue

        # NEW: brightness ratio — void must be brighter than solder mean
        void_mask_local = np.zeros(roi_gray.shape, dtype=np.uint8)
        cv2.drawContours(void_mask_local, [c], -1, 255, -1)
        void_pixels = roi_gray[void_mask_local == 255]
        if void_pixels.size > 0:
            void_mean = float(void_pixels.mean())
            if solder_mean > 0 and (void_mean / solder_mean) < min_brt_ratio:
                continue

        c_shifted = c + [x0, y0]
        voids.append(c_shifted)

    return voids


# ---------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------

def draw_results(image, classified_solders, voids):
    output = image.copy()
    solder_contours = [contour for contour, _ in classified_solders]
    cv2.drawContours(output, solder_contours, -1, SOLDER_COLOR, 2)
    cv2.drawContours(output, voids, -1, VOID_COLOR, 2)
    return output


def build_void_mask(image_shape, voids):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if voids:
        cv2.drawContours(mask, voids, -1, 255, thickness=-1)
    return mask