#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_K(scene: Path):
    calib = json.loads((scene / "_camera" / "calib.json").read_text())

    if "K_matrix" in calib:
        K = np.asarray(calib["K_matrix"], dtype=np.float64)
    else:
        k = calib["K"]
        K = np.array([
            [float(k["fx"]), 0.0, float(k["cx"])],
            [0.0, float(k["fy"]), float(k["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    return K, calib


def load_T_lidar_to_camera(scene: Path):
    extr = json.loads((scene / "_camera" / "extrinsics_lidar.json").read_text())

    if "T_4x4" in extr:
        T = np.asarray(extr["T_4x4"], dtype=np.float64)
    elif "T" in extr:
        T = np.asarray(extr["T"], dtype=np.float64)
    else:
        R = np.asarray(extr["R"], dtype=np.float64)
        t = np.asarray(extr["t"], dtype=np.float64).reshape(3)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t

    return T, extr


def transform_lidar_to_camera(xyz, T):
    ones = np.ones((len(xyz), 1), dtype=np.float64)
    pts_h = np.concatenate([xyz.astype(np.float64), ones], axis=1)
    cam = (T @ pts_h.T).T[:, :3]
    return cam


def project_cam_points(cam_xyz, K, width, height, z_min=0.05, z_max=80.0):
    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    z = cam_xyz[:, 2]

    valid_z = (z > z_min) & (z < z_max)

    x = x[valid_z]
    y = y[valid_z]
    z = z[valid_z]

    if len(z) == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            0,
        )

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

    return (
        ui[inside],
        vi[inside],
        z[inside].astype(np.float32),
        int(valid_z.sum()),
    )


def build_depth(ui, vi, z, width, height):
    depth = np.zeros((height, width), dtype=np.float32)
    density = np.zeros((height, width), dtype=np.uint16)

    for u, v, zz in zip(ui, vi, z):
        old = depth[v, u]
        if old == 0.0 or zz < old:
            depth[v, u] = zz
        density[v, u] += 1

    valid = (depth > 0).astype(np.uint8)
    return depth, valid, density


def draw_overlay(img, ui, vi, z):
    out = img.copy()

    if len(z) == 0:
        return out

    z_clip = np.clip(z, np.percentile(z, 2), np.percentile(z, 98))
    z_norm = (z_clip - z_clip.min()) / max(float(z_clip.max() - z_clip.min()), 1e-6)

    # ½ü´¦ºì£¬Ô¶´¦À¶
    color_val = (255 * (1.0 - z_norm)).astype(np.uint8)
    colors = cv2.applyColorMap(color_val.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    for u, v, c in zip(ui, vi, colors):
        cv2.circle(out, (int(u), int(v)), 1, tuple(int(x) for x in c.tolist()), -1)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--qa-stride", type=int, default=50)
    ap.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    ap.add_argument("--z-min", type=float, default=0.05)
    ap.add_argument("--z-max", type=float, default=80.0)
    args = ap.parse_args()

    scene = Path(args.scene).expanduser().resolve()

    triad_path = scene / "sync" / "radar_camera_lidar_manifest.csv"
    points_manifest_path = scene / "_lidar" / "points_manifest.csv"

    if not triad_path.exists():
        raise FileNotFoundError(triad_path)
    if not points_manifest_path.exists():
        raise FileNotFoundError(points_manifest_path)

    triad = list(csv.DictReader(open(triad_path)))
    points_manifest = list(csv.DictReader(open(points_manifest_path)))

    K, calib = load_K(scene)
    T, extr = load_T_lidar_to_camera(scene)

    depth_dir = scene / "_camera" / "depth_lidar"
    qa_dir = scene / "_qa" / "lidar_projection"
    depth_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    stats_path = qa_dir / "projection_stats.csv"

    n_total = len(triad)
    if args.max_frames > 0:
        n_total = min(n_total, args.max_frames)

    print("[INFO] scene:", scene)
    print("[INFO] triad rows:", len(triad))
    print("[INFO] point frames:", len(points_manifest))
    print("[INFO] processing frames:", n_total)
    print("[INFO] K:\n", K)
    print("[INFO] T_lidar_to_camera:\n", T)

    stats_rows = []

    for row_i, r in enumerate(triad[:n_total]):
        radar_i = int(r["radar_sorted_i"])
        camera_i = int(r["camera_sorted_j"])
        lidar_j = int(r["lidar_sorted_j"])

        img_path = Path(r["image_path_undist"])
        pt_path = Path(points_manifest[lidar_j]["point_path"])

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {img_path}")

        height, width = img.shape[:2]

        data = np.load(pt_path)
        xyz = data["xyz"]

        cam_xyz = transform_lidar_to_camera(xyz, T)
        ui, vi, z, num_front = project_cam_points(
            cam_xyz, K, width, height, z_min=args.z_min, z_max=args.z_max
        )

        depth, valid, density = build_depth(ui, vi, z, width, height)

        depth_path = depth_dir / f"depth_{radar_i:06d}.npz"
        np.savez_compressed(
            depth_path,
            depth=depth.astype(np.float32),
            valid=valid.astype(np.uint8),
            density=density.astype(np.uint16),
            radar_sorted_i=np.array(radar_i, dtype=np.int32),
            camera_sorted_j=np.array(camera_i, dtype=np.int32),
            lidar_sorted_j=np.array(lidar_j, dtype=np.int32),
            image_path=str(img_path),
            point_path=str(pt_path),
        )

        qa_path = ""
        if row_i % args.qa_stride == 0:
            overlay = draw_overlay(img, ui, vi, z)
            qa_file = qa_dir / f"overlay_{radar_i:06d}.jpg"
            cv2.imwrite(str(qa_file), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            qa_path = str(qa_file)

        num_total = int(len(xyz))
        num_projected = int(len(z))
        front_ratio = num_front / max(num_total, 1)
        projected_ratio = num_projected / max(num_total, 1)

        if len(z) > 0:
            med_depth = float(np.median(z))
            min_depth = float(np.min(z))
            max_depth = float(np.max(z))
        else:
            med_depth = 0.0
            min_depth = 0.0
            max_depth = 0.0

        stats_rows.append([
            radar_i,
            camera_i,
            lidar_j,
            num_total,
            num_front,
            num_projected,
            front_ratio,
            projected_ratio,
            min_depth,
            med_depth,
            max_depth,
            str(depth_path),
            qa_path,
            float(r["delta_camera_minus_radar_ms"]),
            float(r["delta_lidar_minus_radar_ms"]),
        ])

        if row_i % 100 == 0:
            print(
                f"[INFO] {row_i}/{n_total}, radar={radar_i}, "
                f"points={num_total}, front={num_front}, projected={num_projected}, "
                f"ratio={projected_ratio:.4f}"
            )

    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "radar_sorted_i",
            "camera_sorted_j",
            "lidar_sorted_j",
            "num_points",
            "num_front_z",
            "num_projected",
            "front_ratio",
            "projected_ratio",
            "min_depth_m",
            "median_depth_m",
            "max_depth_m",
            "depth_npz_path",
            "qa_overlay_path",
            "delta_camera_minus_radar_ms",
            "delta_lidar_minus_radar_ms",
        ])
        w.writerows(stats_rows)

    projected_ratios = np.array([x[7] for x in stats_rows], dtype=float)
    print("[OK] projection finished")
    print("depth_dir:", depth_dir)
    print("qa_dir:", qa_dir)
    print("stats:", stats_path)
    print("projected ratio mean:", float(projected_ratios.mean()))
    print("projected ratio p50 :", float(np.percentile(projected_ratios, 50)))
    print("projected ratio p95 :", float(np.percentile(projected_ratios, 95)))


if __name__ == "__main__":
    main()
