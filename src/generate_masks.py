import os
import cv2

from config import *
from utils import *


def main():
    create_directory(OUTPUT_DIR)
    masks_dir = os.path.join(OUTPUT_DIR, "masks")
    create_directory(masks_dir)
    for class_name in CLASS_VOID_PARAMS:
        create_directory(os.path.join(masks_dir, class_name))

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ])

    print(f"\nFound {len(image_files)} images.\n")

    for file in image_files:
        image_path = os.path.join(INPUT_DIR, file)
        image = load_image(image_path)
        gray = to_grayscale(image)

        # Stage 1: solders
        solders, solder_binary = segment_solders(gray)

        # Stage 1b: classify each solder's shape so we can use per-class
        # void params and color-code the overlay
        classified_solders = [(c, classify_solder(c)) for c in solders]

        # Stage 2: voids, local to each solder, using that class's params.
        # Kept both flat (for the overlay/combined mask) and grouped by
        # class (for the per-class masks).
        all_voids = []
        voids_by_class = {}
        class_counts = {}
        for contour, class_name in classified_solders:
            void_params = CLASS_VOID_PARAMS.get(class_name)
            voids = find_voids_in_solder(gray, contour, void_params)
            all_voids.extend(voids)
            voids_by_class.setdefault(class_name, []).extend(voids)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        overlay = draw_results(image, classified_solders, all_voids)

        filename = os.path.splitext(file)[0]

        if DEBUG_MODE:
            save_image(os.path.join(OUTPUT_DIR, f"{filename}_solder_binary.png"), solder_binary)

        save_image(os.path.join(OUTPUT_DIR, f"{filename}_voids.png"), overlay)

        # Combined binary void mask (every void, any class, filled white)
        combined_mask = build_void_mask(gray.shape, all_voids)
        save_image(os.path.join(masks_dir, f"{filename}_mask.png"), combined_mask)

        # Per-class binary void masks
        for class_name, class_voids in voids_by_class.items():
            class_mask = build_void_mask(gray.shape, class_voids)
            class_dir = os.path.join(masks_dir, class_name)
            create_directory(class_dir)
            save_image(os.path.join(class_dir, f"{filename}_mask.png"), class_mask)

        class_summary = ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items()))
        print(f"{file}: {len(solders)} solders ({class_summary}), {len(all_voids)} voids")

    print("\nProcessing completed successfully.")


if __name__ == "__main__":
    main()