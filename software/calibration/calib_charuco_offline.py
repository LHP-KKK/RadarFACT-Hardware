#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True, help="Folder of images (jpg/png)")
    ap.add_argument("--pattern", type=str, default="*.jpg,*.png,*.jpeg", help="Comma-separated glob patterns")
    ap.add_argument("--out", type=str, default="camera_intrinsics_charuco.yaml")

    # ---- Your board preset (edit if needed) ----
    ap.add_argument("--dict", type=str, default="DICT_5X5_1000")
    ap.add_argument("--squares", type=int, default=9)
    ap.add_argument("--square_m", type=float, default=0.072)
    ap.add_argument("--marker_m", type=float, default=0.05)

    # filters
    ap.add_argument("--min_charuco", type=int, default=15, help="Minimum charuco corners per frame to accept")
    ap.add_argument("--max_images", type=int, default=0, help="0 means no limit")

    # calibration flags
    ap.add_argument("--rational", action="store_true", help="Use CALIB_RATIONAL_MODEL (recommended for wide lenses)")
    ap.add_argument("--thin_prism", action="store_true", help="Use CALIB_THIN_PRISM_MODEL (use only if needed)")
    ap.add_argument("--fix_k3", action="store_true", help="Fix k3=0 and output only k1,k2,p1,p2 for FAST-Calib compatibility")
    return ap.parse_args()


def get_aruco_dict(dict_name: str):
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"Unknown aruco dict: {dict_name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))


def main():
    args = parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    paths = []
    for pat in patterns:
        paths += sorted(glob.glob(str(img_dir / pat)))
    if args.max_images and args.max_images > 0:
        paths = paths[: args.max_images]

    if len(paths) == 0:
        raise RuntimeError("No images found. Check --img_dir and --pattern")

    aruco_dict = get_aruco_dict(args.dict)
    board = cv2.aruco.CharucoBoard(
        (args.squares, args.squares),
        args.square_m,
        args.marker_m,
        aruco_dict
    )

    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    # For debug stats
    used = 0
    total = 0
    per_img_stats = []

    for p in paths:
        total += 1
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            per_img_stats.append((p, 0, 0, "read_fail"))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w,h)

        corners, ids, _ = aruco_detector.detectMarkers(gray)
        n_markers = 0 if ids is None else len(ids)

        if ids is None or len(ids) == 0:
            per_img_stats.append((p, n_markers, 0, "no_marker"))
            continue

        # interpolate charuco corners
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        n_charuco = 0 if charuco_ids is None else len(charuco_ids)
        if charuco_ids is None or n_charuco < args.min_charuco:
            per_img_stats.append((p, n_markers, n_charuco, "too_few_charuco"))
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        used += 1
        per_img_stats.append((p, n_markers, n_charuco, "OK"))

    print(f"[INFO] Found images: {len(paths)}")
    print(f"[INFO] Valid frames used: {used}")
    print(f"[INFO] Image size: {image_size}")

    # Print a short report (top 100)
    print("\n[INFO] Per-image detection (show up to 100):")
    for row in per_img_stats[:100]:
        print(f"  {Path(row[0]).name:30s} markers={row[1]:2d} charuco={row[2]:2d}  {row[3]}")

    if used < 5:
        raise RuntimeError(
            f"Too few valid frames ({used}). "
            f"Try: (1) take more photos with varying angles/distances, "
            f"(2) lower --min_charuco, or (3) verify square/marker sizes and dictionary."
        )

    # --- calibration ---
    flags = 0
    if args.rational:
        flags |= cv2.CALIB_RATIONAL_MODEL
    if args.thin_prism:
        flags |= cv2.CALIB_THIN_PRISM_MODEL
    if args.fix_k3:
        # 固定 k3=0；同时固定更高阶径向畸变，保持 FAST-Calib 常用的 k1,k2,p1,p2 模型
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-8)

    rms, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria
    )

    print("\n[RESULT]")
    print(f"  RMS reprojection error: {rms:.6f} px")
    print("  K =\n", K)
    D_flat = D.reshape(-1)
    print("  D =\n", D_flat)
    if args.fix_k3:
        print("  D_for_FAST_Calib [k1 k2 p1 p2] =\n", D_flat[:4])

    # save yaml
    out_path = Path(args.out)
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("dict", args.dict)
    fs.write("squares", int(args.squares))
    fs.write("square_m", float(args.square_m))
    fs.write("marker_m", float(args.marker_m))
    fs.write("rms", float(rms))
    fs.write("K", K)
    if args.fix_k3:
        # OpenCV 仍可能返回 [k1,k2,p1,p2,k3]，其中 k3 被固定为 0。
        # 为了直接适配 FAST-Calib，这里额外保存 4 参数版本。
        D4 = D_flat[:4].reshape(1, 4)
        fs.write("D", D4)
        fs.write("D_full_opencv", D)
        fs.write("k1", float(D_flat[0]))
        fs.write("k2", float(D_flat[1]))
        fs.write("p1", float(D_flat[2]))
        fs.write("p2", float(D_flat[3]))
        fs.write("k3_fixed", 0.0)
    else:
        fs.write("D", D)
    fs.release()

    print(f"\n[SAVE] {out_path.resolve()}")


if __name__ == "__main__":
    main()
