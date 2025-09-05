import cv2
import numpy as np

# ---------- Circle fit ----------
def fit_circle_least_squares(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float64)
    x = pts[:, 0]; y = pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx**2 + cy**2)
    return int(cx), int(cy), int(r)

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

def innerdiameter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (tuned for ID detection)
    _, thresh = cv2.threshold(gray, 50, 60, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    cx_img, cy_img = w // 2, h // 2   # image center

    candidates = []

    # Collect all contours enclosing the center
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 80:
            continue
        if cv2.pointPolygonTest(cnt, (cx_img, cy_img), False) >= 0:
            cx, cy, r = fit_circle_least_squares(cnt)
            candidates.append((r, cx, cy, i, cnt))

    contour_img = img.copy()

    # Draw all contours in green
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    if candidates:
        # Pick the **smallest circle** enclosing the image center
        r, cx, cy, idx, cnt = min(candidates, key=lambda x: x[0])

        # Draw innermost contour in RED
        cv2.circle(contour_img, (cx, cy), r, (0, 0, 255), 2)  # Red circle
        cv2.circle(contour_img, (cx, cy), 2, (255, 0, 0), -1) # Blue dot

        print(f"Innermost circle: Contour {idx}, Center=({cx},{cy}), Radius={r}")
    else:
        print("No contour enclosing the image center found.")

    # cv2.namedWindow("All Contours + Innermost in Red", cv2.WINDOW_NORMAL)
    # cv2.imshow("All Contours + Innermost in Red", contour_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cx, cy, r

def calibrate_mm_per_px(img, known_od_mm=280.0, known_id_mm=165.0):
    od = outerdiameter(img)
    idc = innerdiameter(img)

    results = {
        "mm_per_px": None,
        "mm_per_px_od": None,
        "mm_per_px_id": None,
        "od_circle": od,
        "id_circle": idc,
        "warnings": []
    }

    if od is not None:
        cx_o, cy_o, r_o = od
        results["mm_per_px_od"] = 280 / (2.0 * r_o)

    if idc is not None:
        cx_i, cy_i, r_i = idc
        results["mm_per_px_id"] = 165 / (2.0 * r_i)

    scales = [s for s in [results["mm_per_px_od"], results["mm_per_px_id"]] if s]
    if scales:
        results["mm_per_px"] = sum(scales) / len(scales)

    if results["mm_per_px_od"] and results["mm_per_px_id"]:
        diff = abs(results["mm_per_px_od"] - results["mm_per_px_id"])
        avg = results["mm_per_px"]
        pct = (diff / avg) * 100.0 if avg > 0 else 0.0
        if pct > 2.0:
            results["warnings"].append(
                f"OD/ID scale mismatch: {pct:.2f}% "
                f"(OD={results['mm_per_px_od']:.6f}, ID={results['mm_per_px_id']:.6f})"
            )
    return results

def save_calibration(results, filename="calibration.txt"):
    with open(filename, "w") as f:
        f.write("Calibration Results\n")
        f.write("===================\n")
        if results["mm_per_px"] is not None:
            f.write(f"Average mm/px: {results['mm_per_px']:.6f}\n")
        if results["mm_per_px_od"] is not None:
            f.write(f"OD mm/px: {results['mm_per_px_od']:.6f}\n")
        if results["mm_per_px_id"] is not None:
            f.write(f"ID mm/px: {results['mm_per_px_id']:.6f}\n")
        f.write("\nWarnings:\n")
        if results["warnings"]:
            for w in results["warnings"]:
                f.write("- " + w + "\n")
        else:
            f.write("None\n")
# import cv2
# import numpy as np

# # ---------- Circle fit (with subpixel precision) ----------
# def fit_circle_least_squares(cnt):
#     pts = cnt.reshape(-1, 2).astype(np.float64)
#     x = pts[:, 0]; y = pts[:, 1]
#     A = np.column_stack([2*x, 2*y, np.ones_like(x)])
#     b = x**2 + y**2
#     c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#     cx, cy = c[0], c[1]
#     r = np.sqrt(c[2] + cx**2 + cy**2)
#     return float(cx), float(cy), float(r)   # keep subpixel precision

# # ---------- Outer Diameter ----------
# def outerdiameter(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold
#     _, thresh = cv2.threshold(gray, 10, 30, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     circles = []
#     for i, cnt in enumerate(contours):
#         if cv2.contourArea(cnt) < 80:
#             continue
#         cx, cy, r = fit_circle_least_squares(cnt)
#         circles.append((r, cx, cy, i))

#     if len(circles) < 2:
#         print("Not enough circles detected.")
#         return None

#     # Sort circles by radius (largest → smallest)
#     circles.sort(key=lambda x: x[0], reverse=True)

#     # Pick the second largest
#     r, cx, cy, idx = circles[1]
#     print(f"Second largest circle: Contour {idx}, Center=({cx:.2f},{cy:.2f}), Radius={r:.2f}")
#     return cx, cy, r

# # ---------- Inner Diameter ----------
# def innerdiameter(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Threshold (tuned for ID detection)
#     _, thresh = cv2.threshold(gray, 50, 60, cv2.THRESH_BINARY)

#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     h, w = gray.shape
#     cx_img, cy_img = w // 2, h // 2   # image center

#     candidates = []
#     for i, cnt in enumerate(contours):
#         if cv2.contourArea(cnt) < 80:
#             continue
#         if cv2.pointPolygonTest(cnt, (cx_img, cy_img), False) >= 0:
#             cx, cy, r = fit_circle_least_squares(cnt)
#             candidates.append((r, cx, cy, i, cnt))

#     if not candidates:
#         print("No contour enclosing the image center found.")
#         return None

#     # Pick the **smallest circle** enclosing the image center
#     r, cx, cy, idx, cnt = min(candidates, key=lambda x: x[0])
#     print(f"Innermost circle: Contour {idx}, Center=({cx:.2f},{cy:.2f}), Radius={r:.2f}")
#     return cx, cy, r

# # ---------- Calibration ----------



# def calibrate_mm_per_px(img, known_od_mm=280.0, known_id_mm=165.0):
#     od = outerdiameter(img)
#     idc = innerdiameter(img)

#     results = {
#         "mm_per_px": None,
#         "mm_per_px_od": None,
#         "mm_per_px_id": None,
#         "od_circle": od,
#         "id_circle": idc,
#         "warnings": []
#     }

#     if od:
#         cx_o, cy_o, r_o = od
#         results["mm_per_px_od"] = 280 / (2.0 * r_o)
#         print(f"OD radius (px): {r_o:.2f}, scale: {results['mm_per_px_od']:.6f} mm/px")

#     if idc:
#         cx_i, cy_i, r_i = idc
#         results["mm_per_px_id"] = 165 / (2.0 * r_i)
#         print(f"ID radius (px): {r_i:.2f}, scale: {results['mm_per_px_id']:.6f} mm/px")

#     scales = [s for s in [results["mm_per_px_od"], results["mm_per_px_id"]] if s]
#     if scales:
#         results["mm_per_px"] = sum(scales) / len(scales)

#     # Check mismatch
#     if results["mm_per_px_od"] and results["mm_per_px_id"]:
#         diff = abs(results["mm_per_px_od"] - results["mm_per_px_id"])
#         avg = results["mm_per_px"]
#         pct = (diff / avg) * 100.0 if avg > 0 else 0.0
#         if pct > 2.0:
#             results["warnings"].append(
#                 f"OD/ID scale mismatch: {pct:.2f}% "
#                 f"(OD={results['mm_per_px_od']:.6f}, ID={results['mm_per_px_id']:.6f})"
#             )

#     return results


# # ---------- Save Calibration ----------
# def save_calibration(results, filename="calibration.txt"):
#     with open(filename, "w") as f:
#         f.write("Calibration Results\n")
#         f.write("===================\n")
#         if results["mm_per_px"] is not None:
#             f.write(f"Average mm/px: {results['mm_per_px']:.6f}\n")
#         if results["mm_per_px_od"] is not None:
#             f.write(f"OD mm/px: {results['mm_per_px_od']:.6f}\n")
#         if results["mm_per_px_id"] is not None:
#             f.write(f"ID mm/px: {results['mm_per_px_id']:.6f}\n")

#         # Save raw radii too
#         if results["od_circle"]:
#             _, _, r_o = results["od_circle"]
#             f.write(f"OD radius (px): {r_o:.2f}\n")
#         if results["id_circle"]:
#             _, _, r_i = results["id_circle"]
#             f.write(f"ID radius (px): {r_i:.2f}\n")

#         f.write("\nWarnings:\n")
#         if results["warnings"]:
#             for w in results["warnings"]:
#                 f.write("- " + w + "\n")
#         else:
#             f.write("None\n")
