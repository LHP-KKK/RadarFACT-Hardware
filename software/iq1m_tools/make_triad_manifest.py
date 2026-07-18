#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path

import numpy as np


def nearest_indices(ref_ts, query_ts):
    idx = np.searchsorted(ref_ts, query_ts)
    idx = np.clip(idx, 0, len(ref_ts) - 1)

    idx0 = np.clip(idx - 1, 0, len(ref_ts) - 1)
    idx1 = idx

    d0 = np.abs(ref_ts[idx0] - query_ts)
    d1 = np.abs(ref_ts[idx1] - query_ts)

    return np.where(d0 <= d1, idx0, idx1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="IQ1M-like scene root, e.g. ~/iq1m_demo/indoor_forward_02_trim.fwd")
    args = ap.parse_args()

    scene = Path(args.scene).expanduser().resolve()

    radar_ts = np.fromfile(scene / "radar" / "ts", dtype="<f8")
    camera_ts = np.fromfile(scene / "_camera" / "ts", dtype="<f8")
    lidar_ts = np.fromfile(scene / "lidar" / "ts", dtype="<f8")

    nn_manifest_path = scene / "sync" / "nn_manifest.csv"
    nn_rows = list(csv.DictReader(open(nn_manifest_path)))

    if len(nn_rows) != len(radar_ts):
        raise RuntimeError(
            f"nn_manifest rows {len(nn_rows)} != radar frames {len(radar_ts)}"
        )

    cam_idx = nearest_indices(camera_ts, radar_ts)
    lidar_idx = nearest_indices(lidar_ts, radar_ts)

    out_path = scene / "sync" / "radar_camera_lidar_manifest.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "radar_sorted_i",
            "radar_orig_idx",
            "radar_sec",

            "camera_sorted_j",
            "camera_orig_idx",
            "camera_sec",
            "delta_camera_minus_radar_sec",
            "delta_camera_minus_radar_ms",

            "lidar_sorted_j",
            "lidar_orig_idx",
            "lidar_sec",
            "delta_lidar_minus_radar_sec",
            "delta_lidar_minus_radar_ms",

            "image_path_undist",
            "trace_root",
            "camera_cache_id",
        ])

        for ri, rts in enumerate(radar_ts):
            cj = int(cam_idx[ri])
            lj = int(lidar_idx[ri])

            image_path = str(scene / "sync" / "aligned_frames_undist" / f"frame_{ri:06d}.jpg")

            w.writerow([
                ri,
                ri,
                float(rts),

                cj,
                cj,
                float(camera_ts[cj]),
                float(camera_ts[cj] - rts),
                float((camera_ts[cj] - rts) * 1000.0),

                lj,
                lj,
                float(lidar_ts[lj]),
                float(lidar_ts[lj] - rts),
                float((lidar_ts[lj] - rts) * 1000.0),

                image_path,
                str(scene),
                f"{scene.name.replace('.', '_')}_frame_{cj:06d}",
            ])

    print("[OK] wrote:", out_path)
    print("radar frames:", len(radar_ts))
    print("camera frames:", len(camera_ts))
    print("lidar frames:", len(lidar_ts))

    cam_dt = camera_ts[cam_idx] - radar_ts
    lidar_dt = lidar_ts[lidar_idx] - radar_ts

    print("\nCamera-Radar dt:")
    print("  mean abs ms:", np.mean(np.abs(cam_dt)) * 1000)
    print("  p95  abs ms:", np.percentile(np.abs(cam_dt), 95) * 1000)
    print("  max  abs ms:", np.max(np.abs(cam_dt)) * 1000)

    print("\nLiDAR-Radar dt:")
    print("  mean abs ms:", np.mean(np.abs(lidar_dt)) * 1000)
    print("  p95  abs ms:", np.percentile(np.abs(lidar_dt), 95) * 1000)
    print("  max  abs ms:", np.max(np.abs(lidar_dt)) * 1000)


if __name__ == "__main__":
    main()
