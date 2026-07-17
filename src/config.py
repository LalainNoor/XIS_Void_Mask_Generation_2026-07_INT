# Input / Output Paths
INPUT_DIR = "images/input"
OUTPUT_DIR = "images/output"

# ---------------- Solder (pad/lead) detection ----------------
SOLDER_BLUR_KERNEL = (5, 5)
SOLDER_MIN_AREA = 600
SOLDER_MAX_AREA = 400000  # raised again -- some boards have a large body/ground
                          # pad (~250-310k px) that should still be classed as
                          # solder_large and checked for voids
SOLDER_MORPH_KERNEL = (7, 7)

# ---------------- Void detection (per solder, local) ----------------
# These are the fallback / default params, used for any class not
# listed in CLASS_VOID_PARAMS below.
VOID_MIN_AREA = 3
VOID_MAX_AREA = 900
VOID_MIN_CIRCULARITY = 0.55
VOID_MIN_SOLIDITY = 0.80

# Local contrast enhancement inside each solder ROI before thresholding
CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (4, 4)

# ---------------- Solder classification ----------------
# Every solder contour is bucketed into one of these classes based on
# its area and aspect ratio (long side / short side of minAreaRect).
# Checked top-to-bottom; first match wins.
#
#   solder_long   -> thin elongated bars / traces (very high aspect ratio)
#   solder_large  -> big square/rect pads or IC bodies (large area)
#   solder_middle -> medium pads, in between a lead and a large pad
#   solder        -> default: a regular individual lead/pad
SOLDER_LONG_ASPECT_THRESH = 4.0   # aspect_ratio >= this -> solder_long
SOLDER_LARGE_AREA_THRESH = 15000  # area >= this (and not "long") -> solder_large
SOLDER_MIDDLE_AREA_THRESH = 3000  # area >= this -> solder_middle

# Per-class void detection params. Any class/key omitted here falls back
# to the VOID_* defaults above. Tune these independently per class.
CLASS_VOID_PARAMS = {
    "solder": {
        "min_area": VOID_MIN_AREA,
        "max_area": VOID_MAX_AREA,
        "min_circularity": VOID_MIN_CIRCULARITY,
        "min_solidity": VOID_MIN_SOLIDITY,
    },
    "solder_middle": {
        "min_area": 5,
        "max_area": 1400,
        "min_circularity": 0.55,
        "min_solidity": 0.80,
    },
    "solder_large": {
        "min_area": 8,
        "max_area": 4000,
        "min_circularity": 0.50,
        "min_solidity": 0.75,
    },
    "solder_long": {
        "min_area": 3,
        "max_area": 900,
        "min_circularity": 0.55,
        "min_solidity": 0.80,
    },
}

# Single color used to draw every solder's outline, regardless of class
# (class is still tracked internally to pick per-class void params below,
# it's just no longer color-coded in the overlay)
SOLDER_COLOR = (255, 128, 0)  # BGR: orange

# Voids are drawn in this color, filled circle/blob for every detected
# void regardless of size (big or small)
VOID_COLOR = (0, 255, 0)  # BGR: green. Swap to (0, 0, 255) for red.

# Debug
DEBUG_MODE = True