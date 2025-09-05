import cv2

def split_image_middle(img_path, save_prefix="output"):
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not load image.")

    h, w = img.shape[:2]

    # Vertical split (left/right halves)
    mid_x = w // 2
    img1 = img[:, :mid_x]   # Left half
    img2 = img[:, mid_x:]   # Right half

    # Save results
    cv2.imwrite(f"{save_prefix}_left.png", img1)
    cv2.imwrite(f"{save_prefix}_right.png", img2)

    print(f"Image split into two halves and saved as {save_prefix}_left.png and {save_prefix}_right.png")

    return img1, img2


# Example usage
img1, img2 = split_image_middle("/home/sudhanshu/Documents/valeo/variant1/Image_20250823122937847.bmp")
