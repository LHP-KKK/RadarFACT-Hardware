#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def pool_depth(depth, valid, out_h, out_w):
    """
    ½«Ï¡Êè depth map ¾ÛºÏ³É¹Ì¶¨Íø¸ñ region depth¡£

    depth: [H, W], float32, meters
    valid: [H, W], uint8/bool

    Êä³ö£º
      pooled_depth: [out_h, out_w]
      pooled_valid: [out_h, out_w]
      pooled_density: [out_h, out_w]
    """
    h, w = depth.shape
    valid = valid.astype(bool)

    y_edges = np.linspace(0, h, out_h + 1).round().astype(int)
    x_edges = np.linspace(0, w, out_w + 1).round().astype(int)

    pooled = np.zeros((out_h, out_w), dtype=np.float32)
    pooled_valid = np.zeros((out_h, out_w), dtype=np.uint8)
    pooled_density = np.zeros((out_h, out_w), dtype=np.uint16)

    for yy in range(out_h):
        y0, y1 = y_edges[yy], y_edges[yy + 1]

        for xx in range(out_w):
            x0, x1 = x_edges[xx], x_edges[xx + 1]

            d_patch = depth[y0:y1, x0:x1]
            v_patch = valid[y0:y1, x0:x1]

            vals = d_patch[v_patch]
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]

            if len(vals) > 0:
                # ÓÃ median ±È mean ¸ü¿¹ÀëÈºµã
                pooled[yy, xx] = float(np.median(vals))
                pooled_valid[yy, xx] = 1
                pooled_density[yy, xx] = min(len(vals), np.iinfo(np.uint16).max)

    return pooled, pooled_valid, pooled_density


def safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def safe_int(row, key, default=0):
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--filter-camera-ms", type=float, default=50.0)
    ap.add_argument("--filter-lidar-ms", type=float, default=60.0)
    ap.add_argument("--min-projected-ratio", type=float, default=0.03)
    ap.add_argument("--region-h", type=int, default=160)
    ap.add_argument("--region-w", type=int, default=160)
    ap.add_argument("--coarse-h", type=int, default=40)
    ap.add_argument("--coarse-w", type=int, default=40)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    scene = Path(args.scene).expanduser().resolve()

    triad_path = scene / "sync" / "radar_camera_lidar_manifest.csv"
    stats_path = scene / "_qa" / "lidar_projection" / "projection_stats.csv"
    depth_dir = scene / "_camera" / "depth_lidar"

    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")
    if not triad_path.exists():
        raise FileNotFoundError(f"triad manifest not found: {triad_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"projection stats not found: {stats_path}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth dir not found: {depth_dir}")

    out_dir = scene / "_camera_region"
    teacher_dir = scene / "_teacher_targets"
    query_path = scene / "sync" / "camera_query_table.csv"

    if out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_dir)

    if teacher_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(teacher_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    triad_rows = read_csv(triad_path)
    stats_rows = read_csv(stats_path)

    stats_by_radar = {
        safe_int(r, "radar_sorted_i"): r
        for r in stats_rows
    }

    kept = []
    skipped_time = 0
    skipped_projection = 0
    skipped_missing_depth = 0
    skipped_missing_stats = 0
    skipped_empty_depth = 0

    radar_sorted_i_all = []
    camera_sorted_j_all = []
    lidar_sorted_j_all = []
    camera_cache_id_all = []

    depth_region_all = []
    valid_region_all = []
    density_region_all = []

    depth_coarse_all = []
    valid_coarse_all = []

    print("[INFO] scene:", scene)
    print("[INFO] triad rows:", len(triad_rows))
    print("[INFO] stats rows:", len(stats_rows))
    print("[INFO] filter camera ms:", args.filter_camera_ms)
    print("[INFO] filter lidar ms:", args.filter_lidar_ms)
    print("[INFO] min projected ratio:", args.min_projected_ratio)

    for idx, r in enumerate(triad_rows):
        radar_i = safe_int(r, "radar_sorted_i")
        camera_j = safe_int(r, "camera_sorted_j")
        lidar_j = safe_int(r, "lidar_sorted_j")

        cam_dt = abs(safe_float(r, "delta_camera_minus_radar_ms"))
        lidar_dt = abs(safe_float(r, "delta_lidar_minus_radar_ms"))

        if cam_dt > args.filter_camera_ms or lidar_dt > args.filter_lidar_ms:
            skipped_time += 1
            continue

        stat = stats_by_radar.get(radar_i)
        if stat is None:
            skipped_missing_stats += 1
            continue

        projected_ratio = safe_float(stat, "projected_ratio")
        if projected_ratio < args.min_projected_ratio:
            skipped_projection += 1
            continue

        depth_path = Path(stat.get("depth_npz_path", ""))
        if not depth_path.exists():
            # ¶µµ×°´ÃüÃûÕÒ
            depth_path = depth_dir / f"depth_{radar_i:06d}.npz"

        if not depth_path.exists():
            skipped_missing_depth += 1
            continue

        data = np.load(depth_path)
        depth = data["depth"].astype(np.float32)
        valid = data["valid"].astype(np.uint8)

        if valid.sum() == 0:
            skipped_empty_depth += 1
            continue

        d_region, v_region, den_region = pool_depth(
            depth, valid, args.region_h, args.region_w
        )
        d_coarse, v_coarse, _ = pool_depth(
            depth, valid, args.coarse_h, args.coarse_w
        )

        cache_id = f"{scene.name.replace('.', '_')}_radar_{radar_i:06d}"

        l3_path = teacher_dir / f"{cache_id}_l3.npz"
        l1_path = teacher_dir / f"{cache_id}_l1.npz"
        density_path = teacher_dir / f"{cache_id}_density.npz"
        valid_path = teacher_dir / f"{cache_id}_valid.npz"

        np.savez_compressed(
            l3_path,
            depth=d_region.astype(np.float32),
            valid=v_region.astype(np.uint8),
            radar_sorted_i=np.array(radar_i, dtype=np.int32),
            camera_sorted_j=np.array(camera_j, dtype=np.int32),
            lidar_sorted_j=np.array(lidar_j, dtype=np.int32),
        )

        np.savez_compressed(
            l1_path,
            depth=d_coarse.astype(np.float32),
            valid=v_coarse.astype(np.uint8),
            radar_sorted_i=np.array(radar_i, dtype=np.int32),
            camera_sorted_j=np.array(camera_j, dtype=np.int32),
            lidar_sorted_j=np.array(lidar_j, dtype=np.int32),
        )

        np.savez_compressed(
            density_path,
            density=den_region.astype(np.uint16),
            radar_sorted_i=np.array(radar_i, dtype=np.int32),
        )

        np.savez_compressed(
            valid_path,
            valid=v_region.astype(np.uint8),
            radar_sorted_i=np.array(radar_i, dtype=np.int32),
        )

        row = {
            "trace": scene.name,

            "radar_sorted_i": radar_i,
            "radar_orig_idx": r.get("radar_orig_idx", radar_i),
            "radar_sec": r.get("radar_sec", ""),

            "camera_sorted_j": camera_j,
            "camera_orig_idx": r.get("camera_orig_idx", camera_j),
            "camera_sec": r.get("camera_sec", ""),

            "lidar_sorted_j": lidar_j,
            "lidar_orig_idx": r.get("lidar_orig_idx", lidar_j),
            "lidar_sec": r.get("lidar_sec", ""),

            "delta_camera_minus_radar_ms": r.get("delta_camera_minus_radar_ms", ""),
            "delta_lidar_minus_radar_ms": r.get("delta_lidar_minus_radar_ms", ""),

            "image_path_undist": r.get("image_path_undist", ""),
            "depth_npz_path": str(depth_path),
            "qa_overlay_path": stat.get("qa_overlay_path", ""),

            "num_points": stat.get("num_points", ""),
            "num_front_z": stat.get("num_front_z", ""),
            "num_projected": stat.get("num_projected", ""),
            "front_ratio": stat.get("front_ratio", ""),
            "projected_ratio": stat.get("projected_ratio", ""),
            "median_depth_m": stat.get("median_depth_m", ""),

            "trace_root": str(scene),
            "camera_cache_id": cache_id,

            "camera_npz_path_l3": str(l3_path),
            "camera_npz_path_l1": str(l1_path),
            "camera_npz_path_density": str(density_path),
            "camera_npz_path_valid": str(valid_path),
        }

        kept.append(row)

        radar_sorted_i_all.append(radar_i)
        camera_sorted_j_all.append(camera_j)
        lidar_sorted_j_all.append(lidar_j)
        camera_cache_id_all.append(cache_id)

        depth_region_all.append(d_region)
        valid_region_all.append(v_region)
        density_region_all.append(den_region)

        depth_coarse_all.append(d_coarse)
        valid_coarse_all.append(v_coarse)

        if len(kept) % 100 == 0:
            print("[INFO] kept rows:", len(kept), "latest radar:", radar_i)

    if len(kept) == 0:
        raise RuntimeError(
            "No rows kept. Try loosening --filter-camera-ms, --filter-lidar-ms, or --min-projected-ratio."
        )

    fieldnames = list(kept[0].keys())
    write_csv(query_path, kept, fieldnames)

    cache_path = out_dir / "camera_region.npz"

    np.savez_compressed(
        cache_path,
        radar_sorted_i=np.asarray(radar_sorted_i_all, dtype=np.int32),
        camera_sorted_j=np.asarray(camera_sorted_j_all, dtype=np.int32),
        lidar_sorted_j=np.asarray(lidar_sorted_j_all, dtype=np.int32),
        camera_cache_id=np.asarray(camera_cache_id_all),

        depth_160=np.stack(depth_region_all).astype(np.float32),
        valid_160=np.stack(valid_region_all).astype(np.uint8),
        density_160=np.stack(density_region_all).astype(np.uint16),

        depth_40=np.stack(depth_coarse_all).astype(np.float32),
        valid_40=np.stack(valid_coarse_all).astype(np.uint8),
    )

    meta = {
        "scene": str(scene),
        "triad_manifest": str(triad_path),
        "projection_stats": str(stats_path),
        "depth_dir": str(depth_dir),
        "query_table": str(query_path),
        "cache_path": str(cache_path),
        "teacher_targets_dir": str(teacher_dir),

        "num_rows_total_triad": len(triad_rows),
        "num_rows_projection_stats": len(stats_rows),
        "num_rows_kept": len(kept),

        "skipped_time": skipped_time,
        "skipped_projection": skipped_projection,
        "skipped_missing_stats": skipped_missing_stats,
        "skipped_missing_depth": skipped_missing_depth,
        "skipped_empty_depth": skipped_empty_depth,

        "filter_camera_ms": args.filter_camera_ms,
        "filter_lidar_ms": args.filter_lidar_ms,
        "min_projected_ratio": args.min_projected_ratio,

        "region_shape": [args.region_h, args.region_w],
        "coarse_shape": [args.coarse_h, args.coarse_w],

        "npz_fields": {
            "depth_160": "[N,160,160] float32, median LiDAR depth in each image region",
            "valid_160": "[N,160,160] uint8, whether each region has LiDAR depth",
            "density_160": "[N,160,160] uint16, number of projected depth pixels in each region",
            "depth_40": "[N,40,40] float32, coarse region depth",
            "valid_40": "[N,40,40] uint8, coarse valid mask",
        },

        "note": (
            "This is a local smoke-test camera region cache generated from LiDAR-projected sparse depth. "
            "It is not yet the final model-specific image feature cache."
        ),
    }

    meta_path = out_dir / "camera_region_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("[OK] camera region cache finished")
    print("query:", query_path)
    print("cache:", cache_path)
    print("meta:", meta_path)
    print("teacher_targets:", teacher_dir)
    print()
    print("kept rows:", len(kept))
    print("skipped_time:", skipped_time)
    print("skipped_projection:", skipped_projection)
    print("skipped_missing_stats:", skipped_missing_stats)
    print("skipped_missing_depth:", skipped_missing_depth)
    print("skipped_empty_depth:", skipped_empty_depth)
    print()
    print("valid_160 mean:", float(np.stack(valid_region_all).mean()))
    print("depth_160 nonzero ratio:", float((np.stack(depth_region_all) > 0).mean()))


if __name__ == "__main__":
    main()
