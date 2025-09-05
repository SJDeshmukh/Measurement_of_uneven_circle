import cv2
import numpy as np

def stitch_images(img1_path, img2_path, save_path="stitched.png", mode="vertical"):
    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError("Could not load one of the images.")

    # Make sure sizes match
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    if mode == "vertical":  # left-right stitching
        if h1 != h2:
            # Resize second image to match first height
            img2 = cv2.resize(img2, (w2, h1))
        stitched = np.hstack((img1, img2))

    elif mode == "horizontal":  # top-bottom stitching
        if w1 != w2:
            # Resize second image to match first width
            img2 = cv2.resize(img2, (w1, h2))
        stitched = np.vstack((img1, img2))
    else:
        raise ValueError("Mode must be 'vertical' or 'horizontal'.")

    # Save result
    cv2.imwrite(save_path, stitched)
    print(f"Stitched image saved as {save_path}")

    return stitched


# Example usage:
stitched = stitch_images("output_left.png", "output_right.png", save_path="stitched.png", mode="vertical")
