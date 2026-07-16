import os

from config import *
from utils import *


def main():

    create_directory(OUTPUT_DIR)

    image_files = sorted([
        file for file in os.listdir(INPUT_DIR)
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
    ])

    print(f"\nFound {len(image_files)} images.\n")

    for idx, file in enumerate(image_files):

        image_path = os.path.join(INPUT_DIR, file)

        # Load image
        image = load_image(image_path)

        # Preprocess image
        gray, clahe, blurred = preprocess_image(image)

        # Find all contours
        binary, contours, hierarchy = find_void_candidates(blurred)

        # Keep only child contours
        child_contours = filter_child_contours(
            contours,
            hierarchy
        )

        # Remove tiny contours
        area_filtered_contours = filter_by_area(
            child_contours
        )

        # Measure contour properties
        properties = measure_contour_properties(
            gray,
            area_filtered_contours
        )
        """
        if idx == 0:

            print("\nMean Intensities")

            for p in properties:

                print(
                    f"ID:{p['id']:2d} | "
                    f"Intensity:{p['mean_intensity']:.1f}"
                )
            """
        
        shape_filtered = filter_by_shape(
            properties
        )

        final_contours = [
            p["contour"]
            for p in shape_filtered
        ]

        # Draw final contours
        overlay = draw_contours(
            image.copy(),
            final_contours
        )

        filename = os.path.splitext(file)[0]

        # Save intermediate images
        save_image(
            os.path.join(OUTPUT_DIR, f"{filename}_gray.png"),
            gray
        )

        save_image(
            os.path.join(OUTPUT_DIR, f"{filename}_clahe.png"),
            clahe
        )

        save_image(
            os.path.join(OUTPUT_DIR, f"{filename}_blur.png"),
            blurred
        )

        save_image(
            os.path.join(OUTPUT_DIR, f"{filename}_binary.png"),
            binary
        )

        # Save final overlay
        save_image(
            os.path.join(OUTPUT_DIR, f"{filename}_voids.png"),
            overlay
        )

        # Print pipeline statistics
        print(
            f"{file} -> "
            f"{len(contours)} total | "
            f"{len(child_contours)} child | "
            f"{len(area_filtered_contours)} area | "
            f"{len(shape_filtered)} shape"
        )

        # Uncomment only when debugging contour measurements
        """
        if idx == 0:

            print("\nContour Properties")

            for p in properties:

                print(
                    f"ID:{p['id']:2d} | "
                    f"Area:{p['area']:.1f} | "
                    f"Circularity:{p['circularity']:.2f} | "
                    f"Solidity:{p['solidity']:.2f} | "
                    f"Aspect:{p['aspect_ratio']:.2f} | "
                    f"Intensity:{p['mean_intensity']:.1f}"
                )
        """

    print("\nProcessing completed successfully.")


if __name__ == "__main__":
    main()