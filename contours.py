
# import cv2
# import numpy as np

# def fit_circle_least_squares(cnt):
#     pts = cnt.reshape(-1, 2).astype(np.float64)
#     x = pts[:, 0]; y = pts[:, 1]
#     A = np.column_stack([2*x, 2*y, np.ones_like(x)])
#     b = x**2 + y**2
#     c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#     cx, cy = c[0], c[1]
#     r = np.sqrt(c[2] + cx**2 + cy**2)
#     return int(cx), int(cy), int(r)

# # ---------- Load image ----------
# img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250828141002341.bmp")
# if img is None:
#     raise FileNotFoundError("Image not found")

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # ---------- Threshold (invert: white holes become foreground) ----------
# _, thresh = cv2.threshold(gray, 80, 100, cv2.THRESH_BINARY)  # For white blobs

# # ---------- Morphological cleanup ----------
# kernel = np.ones((3,3), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# # ---------- Find contours ----------
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# circle_img = img.copy()
# holes_detected = []

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area < 100:  # Ignore tiny noise
#         continue

#     # Check circularity = 4π * Area / Perimeter²
#     perimeter = cv2.arcLength(cnt, True)
#     if perimeter == 0:
#         continue
#     circularity = 4 * np.pi * area / (perimeter * perimeter)

#     if circularity < 0.7:  # You can tweak this
#         continue

#     # Fit circle
#     cx, cy, r = fit_circle_least_squares(cnt)

#     if r <= 1 or r > max(img.shape[:2]) :  # Ignore huge/invalid ones
#         continue

#     holes_detected.append((cx, cy, r))

#     # Draw circle
#     cv2.circle(circle_img, (cx, cy), r, (0, 255, 0), 2)
#     cv2.circle(circle_img, (cx, cy), 2, (0, 255, 255), -1)

# print(f"Detected {len(holes_detected)} circular holes.")

# # ---------- Display ----------
# cv2.namedWindow("Detected Holes", cv2.WINDOW_NORMAL)
# cv2.imshow("Detected Holes", circle_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def fit_circle_least_squares(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return int(cx), int(cy), int(r)

def detect_largest_black_circle(img):
    # img = cv2.imread(image_path)
    print("Inside")
    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to detect dark (black) regions
    _, thresh = cv2.threshold(gray, 140, 200, cv2.THRESH_BINARY_INV)
    cv2.imwrite("thresh.jpg",thresh)
    # Morphological clean-up (optional)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_circle = None
    largest_area = 0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.7:
            continue

        cx, cy, r = fit_circle_least_squares(cnt)
        if r <= 5:
            continue

        if area > largest_area:
            largest_area = area
            largest_circle = (cx, cy, r)
            best_cnt = cnt
    # print(2*r)
    # return largest_circle,img,best_cnt
   
    return largest_circle

# # ---------- Run and Show ----------
# image_path = "/opt/MVS/bin/Temp/Data/Image_20250829101508921.bmp"
# result, img, cnt = detect_largest_black_circle(image_path)

# if result:
#     cx, cy, r = result
#     print(f"Largest black circle -> Center: ({cx}, {cy}), Radius: {r}")
#     # print("Diameter:",2*r*0.145379)
#     cv2.circle(img, (cx, cy), r, (0, 0, 255), 6)  # Red circle
#     cv2.circle(img, (cx, cy), 4, (255, 0, 0), 3)  # Blue center
#     if cnt is not None:
#         cv2.drawContours(img, [cnt], -1, (0, 255, 0), 1)

#     cv2.namedWindow("Largest Black Circle", cv2.WINDOW_NORMAL)
#     cv2.imshow("Largest Black Circle", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
    # print("No black circle found.")
