INPUT_DIR = "images/input"
OUTPUT_DIR = "images/output"

SOLDER_BLUR_KERNEL = (5, 5)
SOLDER_MIN_AREA = 600
SOLDER_MAX_AREA = 400000
SOLDER_MORPH_KERNEL = (7, 7)
BORDER_MARGIN = 20
BORDER_SPAN_FRACTION = 0.5
SOLDER_MIN_SOLIDITY = 0.68
SOLDER_MAX_MEAN_BRIGHTNESS = 200
SOLDER_BRIGHT_PIXEL_RATIO = 0.25
SOLDER_MIN_AREA_BORDER_TOUCH = 12000
SOLDER_MAX_ASPECT_BORDER = 3.5
SOLDER_MIN_CIRCULARITY_BORDER = 0.45

VOID_MIN_AREA = 4
VOID_MAX_AREA = 900
VOID_MIN_CIRCULARITY = 0.60
VOID_MIN_SOLIDITY = 0.80
VOID_MAX_ASPECT_RATIO = 3.0
VOID_MIN_BRIGHTNESS_RATIO = 1.05

CLAHE_CLIP_LIMIT = 3.0
CLAHE_GRID_SIZE = (4, 4)

SOLDER_LONG_ASPECT_THRESH = 4.0
SOLDER_LARGE_AREA_THRESH = 15000
SOLDER_MIDDLE_AREA_THRESH = 3000

CLASS_VOID_PARAMS = {
    "solder": {
        "min_area": 4,
        "max_area": 8000,
        "min_circularity": 0.50,
        "min_solidity": 0.72,
        "max_aspect_ratio": 2.8,
        "min_brightness_ratio": 1.06,
        "use_tophat": True,
    },
    "solder_middle": {
        "min_area": 5,
        "max_area": 15000,
        "min_circularity": 0.42,
        "min_solidity": 0.62,
        "max_aspect_ratio": 3.0,
        "min_brightness_ratio": 1.05,
        "use_tophat": True,
    },
    "solder_large": {
        "min_area": 25,          # raised — kills tiny speckle dots on large pads
        "max_area": 120000,
        "min_circularity": 0.38,
        "min_solidity": 0.58,
        "max_aspect_ratio": 4.0,
        "min_brightness_ratio": 1.04,
        "use_tophat": False,     # OFF — top-hat was picking up gradient artifacts on big pads
    },
    "solder_long": {
        "min_area": 5,
        "max_area": 50000,
        "min_circularity": 0.40,
        "min_solidity": 0.60,
        "max_aspect_ratio": 3.5,
        "min_brightness_ratio": 1.04,
        "use_tophat": False,     # OFF — same issue on long connector pads
    },
}

SOLDER_COLOR = (255, 128, 0)
VOID_COLOR   = (0, 255, 0)
DEBUG_MODE   = True