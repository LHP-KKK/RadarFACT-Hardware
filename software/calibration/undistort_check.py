#!/usr/bin/env python3
"""Visualize camera undistortion and a pixel-difference heat map."""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--images", required=True, help="Image path or glob")
    parser.add_argument("--output-dir", type=Path, default=Path("undistort_vis"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(args.calibration), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(args.calibration)
    camera_matrix = fs.getNode("K").mat()
    distortion = fs.getNode("D").mat()
    fs.release()
    if camera_matrix is None or distortion is None:
        raise ValueError("Calibration file does not contain K and D matrices")

    paths = [Path(item) for item in sorted(glob.glob(args.images))[: args.limit]]
    if not paths:
        raise FileNotFoundError(f"No images matched: {args.images}")

    for path in paths:
        original = cv2.imread(str(path))
        if original is None:
            print(f"[WARN] could not read {path}")
            continue
        height, width = original.shape[:2]
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion, (width, height), args.alpha
        )
        corrected = cv2.undistort(original, camera_matrix, distortion, None, new_matrix)
        difference = cv2.absdiff(original, corrected)
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        stem = path.stem
        cv2.imwrite(str(args.output_dir / f"{stem}_orig.jpg"), original)
        cv2.imwrite(str(args.output_dir / f"{stem}_und.jpg"), corrected)
        cv2.imwrite(str(args.output_dir / f"{stem}_heatmap_overlay.jpg"), overlay)
        cv2.imwrite(
            str(args.output_dir / f"{stem}_comparison.jpg"),
            np.hstack((original, corrected, heatmap)),
        )
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()

