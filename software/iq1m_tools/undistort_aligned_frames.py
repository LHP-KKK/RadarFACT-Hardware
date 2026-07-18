#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def load_cam_json(scene):
    raw_path = scene / "_camera" / "calib_raw.json"
    if not raw_path.exists():
        raw_path = scene / "_camera" / "calib.json"
    if not raw_path.exists():
        raise FileNotFoundError("Cannot find _camera/calib_raw.json or _camera/calib.json")
    return raw_path, json.loads(raw_path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="cv2.getOptimalNewCameraMatrix alpha: 0 removes black border, 1 keeps all pixels")
    args = ap.parse_args()

    scene = Path(args.scene).expanduser().resolve()
    sync_dir = scene / "sync"
    cam_dir = scene / "_camera"
    cam_pub_dir = scene / "camera"
    cam_dir.mkdir(parents=True, exist_ok=True)
    cam_pub_dir.mkdir(parents=True, exist_ok=True)

    src = sync_dir / "aligned_frames_undist"
    backup = sync_dir / "aligned_frames_rawcopy"
    dst = sync_dir / "aligned_frames_undist"

    if not src.exists() and not backup.exists():
        raise FileNotFoundError(f"Cannot find source frames: {src} or {backup}")

    # µÚÒ»´ÎÔËÐÐ£º°Ñ exporter Éú³ÉµÄ raw-copy aligned_frames_undist ±¸·ÝÆðÀ´
    if not backup.exists():
        print("[INFO] backup raw-copy frames:")
        print(" ", src, "->", backup)
        src.rename(backup)
    else:
        print("[INFO] using existing raw-copy backup:", backup)

    # Çå¿Õ²¢ÖØ½¨ undist Êä³öÄ¿Â¼
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    raw_calib_path, cam = load_cam_json(scene)

    width = int(cam["image_width"])
    height = int(cam["image_height"])

    K_info = cam["K"]
    K = np.array([
        [float(K_info["fx"]), 0.0, float(K_info["cx"])],
        [0.0, float(K_info["fy"]), float(K_info["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    D = np.array(cam["distortion"]["coefficients"], dtype=np.float64).reshape(-1)

    newK, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (width, height), args.alpha, (width, height)
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, newK, (width, height), cv2.CV_16SC2
    )

    images = sorted(backup.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No jpg images found in {backup}")

    print("[INFO] undistorting images:", len(images))
    for i, p in enumerate(images):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("[WARN] failed to read:", p)
            continue

        und = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        out = dst / p.name
        ok = cv2.imwrite(str(out), und, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
        if not ok:
            raise RuntimeError(f"Failed to write {out}")

        if i % 200 == 0:
            print(f"[INFO] {i}/{len(images)}")

    undist_cam_json = dict(cam)
    undist_cam_json["K_raw_matrix"] = K.tolist()
    undist_cam_json["K_matrix"] = newK.tolist()
    undist_cam_json["K"] = {
        "fx": float(newK[0, 0]),
        "fy": float(newK[1, 1]),
        "cx": float(newK[0, 2]),
        "cy": float(newK[1, 2]),
    }
    undist_cam_json["distortion_raw"] = cam["distortion"]
    undist_cam_json["distortion"] = {
        "model": "none",
        "coefficients": [0.0, 0.0, 0.0, 0.0],
        "order": ["k1", "k2", "p1", "p2"],
    }
    undist_cam_json["undistortion_applied"] = True
    undist_cam_json["undistorted_image_size"] = [width, height]
    undist_cam_json["undistortion_alpha"] = args.alpha
    undist_cam_json["roi"] = [int(x) for x in roi]
    undist_cam_json["source_rawcopy_dir"] = str(backup)
    undist_cam_json["undistorted_dir"] = str(dst)
    undist_cam_json["notes"] = (
        "Images in sync/aligned_frames_undist are truly undistorted by OpenCV. "
        "Use this updated K for image projection."
    )

    for out_json in [cam_dir / "calib.json", cam_pub_dir / "calib.json"]:
        out_json.write_text(json.dumps(undist_cam_json, indent=2, ensure_ascii=False))

    # Ò²Ð´Ò»·Ý¸üÃ÷È·µÄÎÄ¼þÃû£¬±ãÓÚºóÐø×·×Ù
    (cam_dir / "calib_undistorted.json").write_text(
        json.dumps(undist_cam_json, indent=2, ensure_ascii=False)
    )

    print("[OK] undistortion finished")
    print("raw calib:", raw_calib_path)
    print("raw-copy backup:", backup)
    print("undistorted:", dst)
    print("updated:", cam_dir / "calib.json")
    print("new K:")
    print(newK)


if __name__ == "__main__":
    main()
