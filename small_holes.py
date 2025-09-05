import cv2
import numpy as np

def fit_circle_least_squares(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return int(cx), int(cy), int(r)

def detect_small_holes(img, area_thresh=100, circularity_thresh=0.7, max_radius=100):
    """
    Detects small circular holes in a black disk with white holes.

    Parameters:
        img (numpy.ndarray): Input image (BGR).
        area_thresh (float): Minimum contour area to consider.
        circularity_thresh (float): Minimum circularity (0.0 to 1.0) to consider.
        max_radius (int): Maximum radius for detected holes.

    Returns:
        List of (cx, cy, r) for each valid small hole.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get white holes as foreground
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    holes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_thresh:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < circularity_thresh:
            continue

        cx, cy, r = fit_circle_least_squares(cnt)
        if r <= 1 or r > max_radius:
            continue

        holes.append((cx, cy, r))

    return holes


# img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250828141002341.bmp")
# holes = detect_small_holes(img)

# # print(f"Detected {len(holes)} small holes.")
# # for i, (cx, cy, r) in enumerate(holes):
# #     print(f"Hole {i+1}: Center=({cx}, {cy}), Radius={r}")
# for cx, cy, r in holes:
#     cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
#     cv2.circle(img, (cx, cy), 2, (0, 255, 255), -1)
# cv2.namedWindow("Detected Small Holes", cv2.WINDOW_NORMAL)

# cv2.imshow("Detected Small Holes", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
