# Input / Output Paths
INPUT_DIR = "images/input"
OUTPUT_DIR = "images/output"

# Image Processing Parameters
GAUSSIAN_KERNEL = (5, 5)

# CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# Contour Filtering
MIN_CONTOUR_AREA = 30
MAX_CONTOUR_AREA = 3000

DEBUG_IMAGE_INDEX = 0  # Set to -1 to disable debug output

# Shape Filtering
MIN_CIRCULARITY = 0.75

MIN_SOLIDITY = 0.90

MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.80

# Intensity Filtering
#MAX_MEAN_INTENSITY = 115
#MIN_MEAN_INTENSITY = 0