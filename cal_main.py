import cv2
from calibration import calibrate_mm_per_px, save_calibration

# --- Main ---
img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250823132407836.bmp")
out = img.copy()

# Run calibration
calib = calibrate_mm_per_px(img, known_od_mm=250.0, known_id_mm=150.0)
save_calibration(calib, "calibration.txt")

# Draw OD and ID circles
if calib["od_circle"] is not None:
    cx, cy, r = calib["od_circle"]
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 0), 2)  # OD green
if calib["id_circle"] is not None:
    cx, cy, r = calib["id_circle"]
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 0, 255), 2)  # ID red

# Show results
cv2.namedWindow("Calibration (OD+ID)", cv2.WINDOW_NORMAL)
cv2.imshow("Calibration (OD+ID)", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
