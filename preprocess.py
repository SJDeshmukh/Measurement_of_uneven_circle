
# import cv2
# import numpy as np

# def fit_circle_least_squares(cnt):
#     pts = cnt.reshape(-1, 2).astype(np.float64)
#     x = pts[:, 0]
#     y = pts[:, 1]

#     A = np.column_stack([2*x, 2*y, np.ones_like(x)])
#     b = x**2 + y**2
#     c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

#     cx, cy = c[0], c[1]
#     r = np.sqrt(c[2] + cx**2 + cy**2)
#     return int(cx), int(cy), int(r)



# def outerdiameter(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold
#     _, thresh = cv2.threshold(gray, 10, 30, cv2.THRESH_BINARY)

#     # Find contours
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     contour_img = img.copy()
#     circles = []

#     for i, cnt in enumerate(contours):
#         if cv2.contourArea(cnt) < 80:
#             continue

#         cx, cy, r = fit_circle_least_squares(cnt)
#         circles.append((r, cx, cy, i))

#     if len(circles) < 2:
#         print("Not enough circles detected.")
#         return

#     # Sort circles by radius (largest → smallest)
#     circles.sort(key=lambda x: x[0], reverse=True)

#     # Pick the second largest
#     r, cx, cy, idx = circles[1]

#     # Draw the second largest circle
#     cv2.circle(contour_img, (cx, cy), r, (255, 0, 0), 2)
#     cv2.circle(contour_img, (cx, cy), 2, (0, 0, 255), -1)

#     print(f"Second largest circle: Contour {idx}, Center=({cx},{cy}), Radius={r}")

#     # cv2.namedWindow("Second Largest Circle", cv2.WINDOW_NORMAL)
#     # cv2.imshow("Second Largest Circle", contour_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return cx, cy, r


# def innerdiameter(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold (tuned for ID detection)
#     _, thresh = cv2.threshold(gray, 50, 60, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     h, w = gray.shape
#     cx_img, cy_img = w // 2, h // 2   # image center

#     candidates = []

#     # Collect all contours enclosing the center
#     for i, cnt in enumerate(contours):
#         if cv2.contourArea(cnt) < 80:
#             continue
#         if cv2.pointPolygonTest(cnt, (cx_img, cy_img), False) >= 0:
#             cx, cy, r = fit_circle_least_squares(cnt)
#             candidates.append((r, cx, cy, i, cnt))

#     contour_img = img.copy()

#     # Draw all contours in green
#     cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

#     if candidates:
#         # Pick the **smallest circle** enclosing the image center
#         r, cx, cy, idx, cnt = min(candidates, key=lambda x: x[0])

#         # Draw innermost contour in RED
#         cv2.circle(contour_img, (cx, cy), r, (0, 0, 255), 2)  # Red circle
#         cv2.circle(contour_img, (cx, cy), 2, (255, 0, 0), -1) # Blue dot

#         print(f"Innermost circle: Contour {idx}, Center=({cx},{cy}), Radius={r}")
#     else:
#         print("No contour enclosing the image center found.")

#     # cv2.namedWindow("All Contours + Innermost in Red", cv2.WINDOW_NORMAL)
#     # cv2.imshow("All Contours + Innermost in Red", contour_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return cx, cy, r






# # --- Main ---
# # img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250823102457152.bmp")
# # outerdiameter(img)

# # --- Main ---
# img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250823102457152.bmp")
# draw_img = img.copy()

# od = outerdiameter(img)
# idc = innerdiameter(img)

# # Draw both on same image
# if od:
#     cx, cy, r = od
#     cv2.circle(draw_img, (cx, cy), r, (0, 255, 0), 2)   # Green = OD
#     cv2.circle(draw_img, (cx, cy), 2, (0, 255, 0), -1)

# if idc:
#     cx, cy, r = idc
#     cv2.circle(draw_img, (cx, cy), r, (0, 0, 255), 2)   # Red = ID
#     cv2.circle(draw_img, (cx, cy), 2, (255, 0, 0), -1)

# cv2.namedWindow("OD + ID", cv2.WINDOW_NORMAL)
# cv2.imshow("OD + ID", draw_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# ---------------- Circle fitting ----------------
def fit_circle_least_squares(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return int(cx), int(cy), int(r)

# ---------------- OD ----------------
def outerdiameter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold
    _, thresh = cv2.threshold(gray, 10, 30, cv2.THRESH_BINARY)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    circles = []

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 80:
            continue

        cx, cy, r = fit_circle_least_squares(cnt)
        circles.append((r, cx, cy, i))

    if len(circles) < 2:
        print("Not enough circles detected.")
        return

    # Sort circles by radius (largest → smallest)
    circles.sort(key=lambda x: x[0], reverse=True)

    # Pick the second largest
    r, cx, cy, idx = circles[1]

    # Draw the second largest circle
    cv2.circle(contour_img, (cx, cy), r, (255, 0, 0), 2)
    cv2.circle(contour_img, (cx, cy), 2, (0, 0, 255), -1)

    print(f"Second largest circle: Contour {idx}, Center=({cx},{cy}), Radius={r}")

    # cv2.namedWindow("Second Largest Circle", cv2.WINDOW_NORMAL)
    # cv2.imshow("Second Largest Circle", contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cx, cy, r

# ---------------- ID ----------------
def innerdiameter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 60, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    cx_img, cy_img = w // 2, h // 2

    candidates = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 80:
            continue
        if cv2.pointPolygonTest(cnt, (cx_img, cy_img), False) >= 0:
            cx, cy, r = fit_circle_least_squares(cnt)
            candidates.append((r, cx, cy, i, cnt))

    if not candidates:
        print("No contour enclosing the image center found.")
        return None
    r, cx, cy, _, _ = min(candidates, key=lambda x: x[0])
    return cx, cy, r
/home/sudhanshu/Documents/valeo/Data/preprocess.py
# ---------------- Calibration loader ----------------
def load_calibration(filename="calibration.txt"):
    od_scale = None
    id_scale = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("OD mm/px:"):
                od_scale = float(line.split(":")[1].strip())
            if line.startswith("ID mm/px:"):
                id_scale = float(line.split(":")[1].strip())
    return od_scale, id_scale

# ---------------- Main ----------------
if __name__ == "__main__":
    img = cv2.imread("Image_20250823113259359.bmp")
    draw_img = img.copy()

    # Load calibration factors
    od_scale, id_scale = load_calibration("calibration.txt")

    # Detect circles
    od = outerdiameter(img)
    idc = innerdiameter(img)

    # Draw OD
    if od:
        cx, cy, r = od
        
        cv2.circle(draw_img, (cx, cy), r, (0, 255, 0), 2)   # Green = OD
        cv2.circle(draw_img, (cx, cy), 2, (0, 255, 0), -1)
        if od_scale:
            diameter_mm = (2 * r) * od_scale
            print("OD: ",diameter_mm)
            cv2.putText(draw_img, f"OD: {diameter_mm:.2f} mm",
                        (cx - r, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Draw ID
    if idc:
        cx, cy, r = idc
        cv2.circle(draw_img, (cx, cy), r, (0, 0, 255), 2)   # Red = ID
        cv2.circle(draw_img, (cx, cy), 2, (255, 0, 0), -1)
        if id_scale:
            diameter_mm = (2 * r) * id_scale
            print("ID: ",diameter_mm)
            cv2.putText(draw_img, f"ID: {diameter_mm:.2f} mm",
                        (cx - r, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Show image
    cv2.namedWindow("OD + ID with Diameters", cv2.WINDOW_NORMAL)
    cv2.imshow("OD + ID with Diameters", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
