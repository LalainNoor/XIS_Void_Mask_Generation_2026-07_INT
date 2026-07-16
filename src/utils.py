import os
import cv2
import numpy as np

from config import *

def create_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_image(image_path):
    """Load image in color."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    return image


def to_grayscale(image):
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(gray, kernel_size):
    """Reduce image noise."""
    return cv2.GaussianBlur(gray, kernel_size, 0)


def apply_clahe(gray, clip_limit, grid_size):
    """Enhance local contrast."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=grid_size
    )
    return clahe.apply(gray)


def save_image(path, image):
    cv2.imwrite(path, image)


def preprocess_image(image):
    """
    Complete preprocessing pipeline.
    Returns:
        gray
        clahe
        blurred
    """

    gray = to_grayscale(image)

    clahe = apply_clahe(
        gray,
        CLAHE_CLIP_LIMIT,
        CLAHE_GRID_SIZE
    )

    blurred = apply_gaussian_blur(
        clahe,
        GAUSSIAN_KERNEL
    )

    return gray, clahe, blurred

def find_void_candidates(image):
    """
    Find all potential void contours.

    Parameters:
        image (numpy.ndarray): Preprocessed grayscale image.

    Returns:
        binary (numpy.ndarray): Binary image used for contour detection.
        contours (list): All detected contours.
    """

    # Otsu threshold
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return binary, contours, hierarchy

def draw_contours(image, contours):
    """
    Draw all detected contours.
    """

    output = image.copy()

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 255, 0),
        1
    )

    return output

def filter_by_area(contours):
    """
    Keep contours whose area lies within the configured range.
    """

    filtered = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        if area > MAX_CONTOUR_AREA:
            continue

        filtered.append(contour)

    return filtered

def filter_child_contours(contours, hierarchy):
    """
    Keep only child contours.
    """

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]

    filtered = []

    for i, contour in enumerate(contours):

        # Parent index
        parent = hierarchy[i][3]

        # Keep only contours that have a parent
        if parent != -1:
            filtered.append(contour)

    return filtered

def measure_contour_properties(image, contours):
    """
    Measure geometric and intensity properties of contours.

    Parameters:
        image: Grayscale image.
        contours: List of contours.

    Returns:
        properties: List of dictionaries.
    """

    properties = []

    for i, contour in enumerate(contours):

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        solidity = 0

        if hull_area > 0:
            solidity = area / hull_area

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / h if h > 0 else 0

        # Mask for intensity calculation
        mask = np.zeros(image.shape, dtype=np.uint8)

        cv2.drawContours(
            mask,
            [contour],
            -1,
            255,
            -1
        )

        mean_intensity = cv2.mean(image, mask=mask)[0]

        properties.append({

        "id": i,

        "contour": contour,

        "area": area,

        "perimeter": perimeter,

        "circularity": circularity,

        "solidity": solidity,

        "aspect_ratio": aspect_ratio,

        "mean_intensity": mean_intensity
        
    })
        
    return properties

def filter_by_shape(properties):
    """
    Filter contours using geometric properties.
    """

    filtered = []

    for p in properties:

        if p["circularity"] < MIN_CIRCULARITY:
            continue

        if p["solidity"] < MIN_SOLIDITY:
            continue

        if p["aspect_ratio"] < MIN_ASPECT_RATIO:
            continue

        if p["aspect_ratio"] > MAX_ASPECT_RATIO:
            continue

        filtered.append(p)

    return filtered



