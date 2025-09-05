
import cv2
import numpy as np
from contours import detect_largest_black_circle

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

# ---------------- Small Hole Detector ----------------
def detect_small_holes(img, area_thresh=100, circularity_thresh=0.7, max_radius=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
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

# ---------------- Inner Diameter Detector ----------------
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

# ---------------- Calibration loader ----------------
def load_calibration(filename="calibration_new_setup.txt"):
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
    img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250904094907407.bmp")
    draw_img = img.copy()

    # Load calibration factors
    od_scale, id_scale = load_calibration("calibration_new_setup.txt")
    print("OD Scale:", od_scale, "| ID Scale:", id_scale)

    # Detect OD and ID
    od = detect_largest_black_circle(img)
    idc = innerdiameter(img)

    # Draw OD
    if od:
        cx, cy, r = od
        cv2.circle(draw_img, (cx, cy), r, (0, 255, 0), 2)   # Green = OD
        cv2.circle(draw_img, (cx, cy), 2, (0, 255, 0), -1)
        if od_scale:
            diameter_mm = (2 * r) * od_scale
            print(f"OD: {diameter_mm:.2f} mm")
            cv2.putText(draw_img, f"OD: {diameter_mm:.2f} mm",
                        (cx - r, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw ID
    if idc:
        cx, cy, r = idc
        cv2.circle(draw_img, (cx, cy), r, (0, 0, 255), 2)   # Red = ID
        cv2.circle(draw_img, (cx, cy), 2, (255, 0, 0), -1)
        if id_scale:
            diameter_mm = (2 * r) * id_scale
            print(f"ID: {diameter_mm:.2f} mm")
            cv2.putText(draw_img, f"ID: {diameter_mm:.2f} mm",
                        (cx - r, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Detect and draw small holes
    holes = detect_small_holes(img)
    print(f"Small holes detected: {len(holes)}")

    for cx, cy, r in holes:
        cv2.circle(draw_img, (cx, cy), r, (255, 255, 0), 1)       # Cyan = Small hole
        cv2.circle(draw_img, (cx, cy), 2, (0, 255, 255), -1)      # Yellow center

    # Show and save
    cv2.namedWindow("OD + ID + Small Holes", cv2.WINDOW_NORMAL)
    cv2.imshow("OD + ID + Small Holes", draw_img)
    cv2.imwrite("Result_with_small_holes.png", draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
