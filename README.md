# Solder Void Detection — X-Ray Image Processing Pipeline

Classical computer vision pipeline for detecting voids (air bubbles) in solder joints from X-ray images. Produces annotated overlays and binary void masks per component class.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Configuration](#configuration)
- [Output](#output)
- [Design Decisions & Known Limits](#design-decisions--known-limits)

---

## Overview

In X-ray imaging, voids in solder joints appear as **brighter regions** than the surrounding solder (less material = less X-ray absorption). This pipeline:

1. Segments solder pads/joints from the X-ray background
2. Classifies each solder by shape/size into one of four classes
3. Detects voids inside each solder using per-class tuned parameters
4. Saves annotated overlays and binary void masks

Tested on VISCOM X-ray inspection images across multiple component types: large connector pads, IC packages with wire bonds, and SMD components.

---

## Requirements

```
Python >= 3.8
opencv-python
numpy
```

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

1. Place X-ray images into `images/input/`
2. Run:

```bash
python generate_masks.py
```

Output is written to `images/output/`. Console prints per-image solder count, class breakdown, and void count.

---

## Pipeline

### Stage 1 — Solder Segmentation

Extracts solder pad regions from the X-ray background.

1. Gaussian blur to reduce sensor noise
2. Otsu thresholding (inverted) — solder is darker than background in X-ray
3. Morphological close + open to fill gaps and remove small blobs
4. Contour filtering by:
   - Area (`SOLDER_MIN_AREA` – `SOLDER_MAX_AREA`)
   - Border touch — small/crescent/bright shapes at image edges are rejected (catches partial components, screw holes, logos)
   - Solidity — rejects jagged shapes (text, logos)
   - Mean brightness — rejects bright vias/screw holes

### Stage 1b — Solder Classification

Each detected solder contour is classified into one of four shape classes based on area and aspect ratio of its minimum-area bounding rectangle:

| Class | Criteria | Typical component |
|---|---|---|
| `solder_long` | Aspect ratio ≥ 4.0 | Connector pins, long pads |
| `solder_large` | Area ≥ 15,000 px² | Thermal/ground pads, IC die attach |
| `solder_middle` | Area ≥ 3,000 px² | Mid-size SMD pads |
| `solder` | Everything else | Small SMD pads |

Class determines which void detection parameters are used (see [Configuration](#configuration)).

### Stage 2 — Void Detection

For each solder contour, voids are detected locally:

1. **ROI extraction** — bounding rect of the solder + 2px padding
2. **Contrast enhancement** — CLAHE followed by percentile stretch (2nd–99th)
3. **Mask erosion** — strips 1–2px of the solder edge (adaptive by area) to prevent background bleed
4. **Thresholding** — two passes, combined with bitwise OR:
   - **Pass 1 (Otsu/Triangle)** — global threshold on the CLAHE-stretched ROI. Catches voids clearly brighter than the whole solder body. `solder_large` and `solder_long` use Otsu only (no Pass 2) because their large area makes the top-hat sensitive to X-ray gradient artifacts
   - **Pass 2 (White Top-Hat)** — enabled for `solder` and `solder_middle` only. Detects locally bright voids on darker patches within the same pad that global Otsu would miss
5. **Morphological open** — 3×3 ellipse kernel removes speckle noise
6. **Contour filtering** per void candidate:
   - Area within `[min_area, max_area]`
   - Circularity ≥ `min_circularity` — real voids are roughly circular
   - Solidity ≥ `min_solidity` — rejects jagged/fragmented blobs
   - Aspect ratio ≤ `max_aspect_ratio` — rejects elongated wire-bond shadows and edge streaks
   - Brightness ratio — void region mean must exceed solder mean by `min_brightness_ratio` (X-ray physics: air = brighter)

---

## Configuration

All parameters are in `config.py`.

### Solder Segmentation

| Parameter | Default | Description |
|---|---|---|
| `SOLDER_BLUR_KERNEL` | `(5,5)` | Gaussian blur before thresholding |
| `SOLDER_MIN_AREA` | `600` | Minimum solder contour area (px²) |
| `SOLDER_MAX_AREA` | `400000` | Maximum solder contour area (px²) |
| `SOLDER_MORPH_KERNEL` | `(7,7)` | Morphology kernel for close/open |
| `BORDER_MARGIN` | `20` | Px from image edge to flag border-touch |
| `SOLDER_MIN_AREA_BORDER_TOUCH` | `12000` | Border-touching solders below this are rejected |
| `SOLDER_MIN_CIRCULARITY_BORDER` | `0.45` | Border-touching solders below this circularity are rejected |
| `SOLDER_MAX_ASPECT_BORDER` | `3.5` | Reject flat/crescent border shapes above this aspect ratio |
| `SOLDER_MIN_SOLIDITY` | `0.68` | Rejects jagged shapes (text, logos) |
| `SOLDER_MAX_MEAN_BRIGHTNESS` | `200` | Rejects bright screw holes / vias |
| `SOLDER_BRIGHT_PIXEL_RATIO` | `0.25` | Max fraction of pixels > 200 before rejection |

### Solder Classification Thresholds

| Parameter | Default | Description |
|---|---|---|
| `SOLDER_LONG_ASPECT_THRESH` | `4.0` | Min aspect ratio for `solder_long` class |
| `SOLDER_LARGE_AREA_THRESH` | `15000` | Min area (px²) for `solder_large` class |
| `SOLDER_MIDDLE_AREA_THRESH` | `3000` | Min area (px²) for `solder_middle` class |

### CLAHE

| Parameter | Default | Description |
|---|---|---|
| `CLAHE_CLIP_LIMIT` | `3.0` | CLAHE clip limit |
| `CLAHE_GRID_SIZE` | `(4,4)` | CLAHE tile grid size |

### Per-Class Void Parameters

Defined in `CLASS_VOID_PARAMS` dict in `config.py`. Each class has:

| Key | Description |
|---|---|
| `min_area` | Minimum void area (px²) |
| `max_area` | Maximum void area (px²) |
| `min_circularity` | Minimum circularity `4πA/P²` |
| `min_solidity` | Minimum solidity `area/hull_area` |
| `max_aspect_ratio` | Maximum bounding rect aspect ratio |
| `min_brightness_ratio` | Minimum `void_mean / solder_mean` |
| `use_tophat` | `True` for small pads; `False` for large/long pads |

**Defaults per class:**

| Class | min_area | max_area | min_circ | min_sol | max_aspect | min_brt | use_tophat |
|---|---|---|---|---|---|---|---|
| `solder` | 4 | 8000 | 0.50 | 0.72 | 2.8 | 1.06 | True |
| `solder_middle` | 5 | 15000 | 0.42 | 0.62 | 3.0 | 1.05 | True |
| `solder_large` | 25 | 120000 | 0.38 | 0.58 | 4.0 | 1.04 | False |
| `solder_long` | 5 | 50000 | 0.40 | 0.60 | 3.5 | 1.04 | False |

> **Why `use_tophat=False` for large/long pads?**
> X-ray beam geometry creates slow-varying intensity gradients across large pads (visible as banding). The top-hat transform detects local brightness relative to neighborhood — on gradient backgrounds it picks up the gradient transitions as voids (false positives). Otsu alone is sufficient since real voids on large pads are bright enough to clear the global threshold.

> **Why higher `min_area` (25) for `solder_large`?**
> Large pads have more surface area for noise. Raising `min_area` eliminates sub-pixel speckle that survives morphological cleaning.

---

## Output

### Annotated Overlay (`*_voids.png`)
- **Blue contours** — detected solder pads
- **Green contours** — detected voids

### Binary Void Masks (`masks/*_mask.png`)
- Single-channel 8-bit, same resolution as input
- White (255) = void region, black (0) = background
- Saved both combined (`masks/`) and per-class (`masks/<class>/`)

### Debug Binary (`*_solder_binary.png`)
- Solder segmentation result before contour filtering
- Only saved when `DEBUG_MODE = True` in `config.py`

---

## Design Decisions & Known Limits

**Classical CV ceiling on large connector pads (Images 1 & 5 type):**
The X-ray exposure gradient across very large pads creates local intensity variations that are physically indistinguishable from small voids using pixel intensity alone. Some residual false positives on large pads are unavoidable without a learned model.

**Top-hat disabled for `solder_large` / `solder_long`:**
Tested and confirmed that top-hat causes significant false positives on gradient-heavy large pads without meaningfully improving real void recall (large voids are bright enough for Otsu alone).

**No model training used:**
The entire pipeline is classical CV (thresholding, morphology, contour analysis). All parameters are interpretable and adjustable via `config.py` without retraining.

**Wire bond artifacts (IC packages):**
Wire bond attachment points and shadows in IC X-ray images are suppressed by the `max_aspect_ratio` and `min_brightness_ratio` filters. Remaining wire-bond related detections in individual pads are typically real voids at the bond site.