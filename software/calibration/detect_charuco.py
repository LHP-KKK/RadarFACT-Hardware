#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


def make_charuco_board(squares: int, square_m: float, marker_m: float, dict_name: str):
    aruco = cv2.aruco
    dict_map = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
    }
    if dict_name not in dict_map:
        raise ValueError(f"Unknown dict_name={dict_name}. Options: {list(dict_map.keys())}")

    dictionary = aruco.getPredefinedDictionary(dict_map[dict_name])

    # OpenCV 4.7+ supports cv2.aruco.CharucoBoard((squaresX, squaresY), ...)
    # For older versions, use CharucoBoard_create
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares, squares), square_m, marker_m, dictionary)
    else:
        board = aruco.CharucoBoard_create(squares, squares, square_m, marker_m, dictionary)

    return dictionary, board


def detect_red_ring_ellipse(bgr: np.ndarray):
    """Return ellipse params ((cx,cy),(MA,ma),angle) or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around hue: [0,10] and [170,180]
    lower1 = np.array([0, 80, 80], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 80], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    # Pick largest contour
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None

    if len(cnt) < 5:
        return None

    ellipse = cv2.fitEllipse(cnt)
    return ellipse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to image (png/jpg)")
    parser.add_argument("--dict", type=str, default="DICT_5X5_1000")
    parser.add_argument("--squares", type=int, default=6)
    parser.add_argument("--square_m", type=float, default=0.105, help="Must match your generated board")
    parser.add_argument("--marker_m", type=float, default=0.08, help="Must match your generated board")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory (default: alongside image)")
    parser.add_argument("--detect_ring", action="store_true", help="Also detect red ring ellipse")
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    save_dir = Path(args.save_dir) if args.save_dir else img_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    dictionary, board = make_charuco_board(args.squares, args.square_m, args.marker_m, args.dict)
    aruco = cv2.aruco

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detector API differs across OpenCV versions
    corners, ids, rejected = None, None, None
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)

    vis = bgr.copy()
    n_markers = 0 if ids is None else len(ids)

    if n_markers > 0:
        aruco.drawDetectedMarkers(vis, corners, ids)

    # Interpolate ChArUco corners
    charuco_corners, charuco_ids = None, None
    n_charuco = 0
    if n_markers > 0:
        # returns (retval, charucoCorners, charucoIds)
        retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        if retval is not None and retval > 0 and charuco_corners is not None:
            n_charuco = len(charuco_corners)
            aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, (0, 255, 0))

    # Optional: detect ring ellipse
    ellipse = None
    if args.detect_ring:
        ellipse = detect_red_ring_ellipse(bgr)
        if ellipse is not None:
            cv2.ellipse(vis, ellipse, (255, 0, 0), 3, lineType=cv2.LINE_AA)

    # Report
    print(f"[OK] image: {img_path}")
    print(f"[Aruco] detected markers: {n_markers}")
    if n_markers > 0:
        id_list = ids.flatten().tolist()
        print(f"[Aruco] ids: {sorted(id_list)}")
    print(f"[ChArUco] interpolated corners: {n_charuco}")

    if args.detect_ring:
        if ellipse is None:
            print("[Ring] ellipse: NOT found")
        else:
            (cx, cy), (MA, ma), angle = ellipse
            print(f"[Ring] ellipse center=({cx:.1f},{cy:.1f}), axes=({MA:.1f},{ma:.1f}), angle={angle:.1f}")

    out_png = save_dir / f"{img_path.stem}_detected.png"
    cv2.imwrite(str(out_png), vis)
    print(f"[Saved] {out_png}")


if __name__ == "__main__":
    main()

# python detect_charuco.py \
#   --img out/charuco_with_border_ring_board1p0m_content0p630m_6x6_300dpi.png \
#   --dict DICT_5X5_1000 \
#   --squares 6 \
#   --square_m 0.105 \
#   --marker_m 0.08 \
#   --detect_ring
