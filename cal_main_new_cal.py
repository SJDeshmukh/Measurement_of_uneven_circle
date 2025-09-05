# main_calibration.py

import cv2
from calibration_new_setup import calibrate_mm_per_px, save_calibration
from calibration_new_setup import *
# --- Load image ---
img = cv2.imread("/opt/MVS/bin/Temp/Data/Image_20250904092620398.bmp")
if img is None:
    raise FileNotFoundError("Image could not be loaded.")

out = img.copy()

# --- Run calibration ---
calib = calibrate_mm_per_px(img, known_od_mm=280.0, known_id_mm=165.0)

# --- Save to file ---
save_calibration(calib, "/home/sudhanshu/Documents/valeo/Data/calibration_new_setup.txt")

# --- Visualize Results ---
if calib["od_circle"] is not None:
    cx, cy, r = calib["od_circle"]

    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 255, 0), 2)  # OD in Green

if calib["id_circle"] is not None:
    cx, cy, r = calib["id_circle"]
    cv2.circle(out, (int(cx), int(cy)), int(r), (0, 0, 255), 2)  # ID in Red

cv2.namedWindow("Calibration (OD + ID)", cv2.WINDOW_NORMAL)
cv2.imshow("Calibration (OD + ID)", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
