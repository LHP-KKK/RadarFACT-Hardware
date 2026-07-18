#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


RADAR_SHAPE = (64, 3, 4, 512)
RADAR_DTYPE = np.dtype("<i2")
RADAR_FRAME_BYTES = int(np.prod(RADAR_SHAPE)) * RADAR_DTYPE.itemsize

RAW_LIDAR_FRAME_FIELDS = {
    "ts": np.dtype("<f8"),
    "sensor_ts": np.dtype("<f8"),
    "ts.device_time.backup": np.dtype("<f8"),
    "timebase": np.dtype("<u8"),
    "lidar_id": np.dtype("u1"),
    "point_start": np.dtype("<u8"),
    "point_count": np.dtype("<u4"),
}

RAW_LIDAR_POINT_FIELDS = {
    "xyz": (np.dtype("<f4"), 3),
    "offset_time": (np.dtype("<u4"), 1),
    "reflectivity": (np.dtype("u1"), 1),
    "tag": (np.dtype("u1"), 1),
    "line": (np.dtype("u1"), 1),
}

RAW_LIDAR_REQUIRED = (
    "ts",
    "timebase",
    "lidar_id",
    "point_start",
    "point_count",
    "xyz",
    "offset_time",
    "reflectivity",
    "tag",
    "line",
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_array(path: Path, dtype: np.dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=dtype)


def validate_timestamps(name: str, ts: np.ndarray) -> None:
    if len(ts) == 0:
        raise RuntimeError(f"{name} timestamp stream is empty")
    if not np.isfinite(ts).all():
        raise RuntimeError(f"{name} timestamps contain NaN or Inf")
    backward = np.flatnonzero(np.diff(ts) < 0)
    if len(backward):
        raise RuntimeError(
            f"{name} timestamps are not monotonic; "
            f"first backward index={int(backward[0])}"
        )


def nearest_indices(ref_ts: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    if len(ref_ts) == 0:
        raise RuntimeError("Reference timestamp array is empty")

    idx = np.searchsorted(ref_ts, query_ts)
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    idx0 = np.clip(idx - 1, 0, len(ref_ts) - 1)
    idx1 = idx

    d0 = np.abs(ref_ts[idx0] - query_ts)
    d1 = np.abs(ref_ts[idx1] - query_ts)
    return np.where(d0 <= d1, idx0, idx1)


def is_raw_mid360(lidar_dir: Path) -> bool:
    return all((lidar_dir / name).is_file() for name in RAW_LIDAR_REQUIRED)


def validate_raw_lidar(lidar_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = load_array(lidar_dir / "ts", np.dtype("<f8"))
    starts = load_array(lidar_dir / "point_start", np.dtype("<u8"))
    counts = load_array(lidar_dir / "point_count", np.dtype("<u4"))

    validate_timestamps("lidar", ts)

    n = len(ts)
    for name, arr in (
        ("point_start", starts),
        ("point_count", counts),
        ("timebase", load_array(lidar_dir / "timebase", np.dtype("<u8"))),
        ("lidar_id", load_array(lidar_dir / "lidar_id", np.dtype("u1"))),
    ):
        if len(arr) != n:
            raise RuntimeError(
                f"LiDAR frame field {name} has {len(arr)} entries, expected {n}"
            )

    for optional_name in ("sensor_ts", "ts.device_time.backup"):
        path = lidar_dir / optional_name
        if path.is_file():
            arr = load_array(path, np.dtype("<f8"))
            if len(arr) != n:
                raise RuntimeError(
                    f"LiDAR frame field {optional_name} has {len(arr)} "
                    f"entries, expected {n}"
                )

    if n:
        expected_points = int(starts[-1]) + int(counts[-1])
    else:
        expected_points = 0

    for name, (dtype, width) in RAW_LIDAR_POINT_FIELDS.items():
        path = lidar_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        item_bytes = dtype.itemsize * width
        size = path.stat().st_size
        if size % item_bytes != 0:
            raise RuntimeError(
                f"LiDAR point field {name} has invalid byte size {size}"
            )
        actual_points = size // item_bytes
        if actual_points != expected_points:
            raise RuntimeError(
                f"LiDAR point field {name} contains {actual_points} points, "
                f"expected {expected_points}"
            )

    return ts, starts, counts


def copy_byte_range(
    src_path: Path,
    dst_path: Path,
    start_item: int,
    item_count: int,
    item_bytes: int,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> None:
    offset = start_item * item_bytes
    remaining = item_count * item_bytes

    with src_path.open("rb") as fin, dst_path.open("wb") as fout:
        fin.seek(offset)
        while remaining:
            block = fin.read(min(chunk_bytes, remaining))
            if not block:
                raise RuntimeError(
                    f"Short read while copying {src_path}; "
                    f"{remaining} bytes still expected"
                )
            fout.write(block)
            remaining -= len(block)


def trim_raw_mid360(
    src_lidar: Path,
    dst_lidar: Path,
    keep_lidar: np.ndarray,
) -> np.ndarray:
    src_ts, src_starts, src_counts = validate_raw_lidar(src_lidar)

    if len(keep_lidar) == 0:
        raise RuntimeError("No LiDAR frames inside trim interval")

    # A timestamp interval over monotonic timestamps should produce one
    # contiguous block. Enforce this so point data can be copied efficiently.
    expected = np.arange(keep_lidar[0], keep_lidar[-1] + 1, dtype=np.int64)
    if not np.array_equal(keep_lidar, expected):
        raise RuntimeError("LiDAR keep indices are unexpectedly non-contiguous")

    first_frame = int(keep_lidar[0])
    last_frame = int(keep_lidar[-1])

    first_point = int(src_starts[first_frame])
    total_points = int(
        src_starts[last_frame] + src_counts[last_frame] - first_point
    )

    # Trim every per-frame field that exists.
    for name, dtype in RAW_LIDAR_FRAME_FIELDS.items():
        src_path = src_lidar / name
        if not src_path.is_file():
            continue

        arr = np.fromfile(src_path, dtype=dtype)
        if len(arr) != len(src_ts):
            raise RuntimeError(
                f"LiDAR frame field {name} has {len(arr)} entries, "
                f"expected {len(src_ts)}"
            )

        trimmed = arr[keep_lidar].copy()
        if name == "point_start":
            trimmed = trimmed - np.asarray(first_point, dtype=trimmed.dtype)

        trimmed.astype(dtype, copy=False).tofile(dst_lidar / name)

    # Copy the contiguous point range for all per-point fields.
    for name, (dtype, width) in RAW_LIDAR_POINT_FIELDS.items():
        copy_byte_range(
            src_lidar / name,
            dst_lidar / name,
            start_item=first_point,
            item_count=total_points,
            item_bytes=dtype.itemsize * width,
        )

    trimmed_ts = src_ts[keep_lidar].astype("<f8", copy=False)

    # Final integrity validation on the trimmed output.
    out_ts, out_starts, out_counts = validate_raw_lidar(dst_lidar)
    if len(out_ts) != len(trimmed_ts):
        raise RuntimeError("Trimmed LiDAR frame count validation failed")
    if int(out_starts[0]) != 0:
        raise RuntimeError("Trimmed LiDAR point_start must begin at zero")
    expected_points = int(out_starts[-1] + out_counts[-1])
    if expected_points != total_points:
        raise RuntimeError(
            f"Trimmed LiDAR point count mismatch: "
            f"{expected_points} != {total_points}"
        )

    print("[INFO] full lidar frames:", len(src_ts))
    print("[INFO] keep lidar frames:", len(trimmed_ts))
    print("[INFO] keep lidar points:", total_points)

    return trimmed_ts


def trim_pseudo_lidar(
    src_lidar: Path,
    dst_lidar: Path,
    keep_lidar: np.ndarray,
) -> np.ndarray:
    """Trim the legacy 128x2048 uint16 LZMA stream.

    This path is only for compatibility exports. The whole compressed stream
    is decompressed and rewritten, so raw Mid-360 is preferred.
    """
    ts = load_array(src_lidar / "ts", np.dtype("<f8"))
    trimmed_ts = ts[keep_lidar].astype("<f8", copy=False)
    trimmed_ts.tofile(dst_lidar / "ts")

    rng_path = src_lidar / "rng"
    if not rng_path.is_file():
        raise FileNotFoundError(rng_path)

    frame_values = 128 * 2048
    raw = lzma.open(rng_path, "rb").read()
    values = np.frombuffer(raw, dtype="<u2")

    if len(values) != len(ts) * frame_values:
        raise RuntimeError(
            "Pseudo LiDAR rng length does not match timestamp count"
        )

    frames = values.reshape(len(ts), 128, 2048)
    with lzma.open(dst_lidar / "rng", "wb", format=lzma.FORMAT_XZ) as fout:
        fout.write(frames[keep_lidar].astype("<u2", copy=False).tobytes())

    print("[INFO] full pseudo lidar frames:", len(ts))
    print("[INFO] keep pseudo lidar frames:", len(trimmed_ts))
    return trimmed_ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--margin-sec", type=float, default=0.10)
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"Source scene not found: {src}")
    if src == dst:
        raise RuntimeError("--src and --dst must be different directories")

    src_radar = src / "radar"
    src_camera = src / "_camera"
    src_lidar = src / "lidar"

    radar_ts_full = load_array(src_radar / "ts", np.dtype("<f8"))
    radar_valid_full = load_array(src_radar / "valid", np.dtype("u1"))
    camera_ts = load_array(src_camera / "ts", np.dtype("<f8"))
    lidar_ts_full = load_array(src_lidar / "ts", np.dtype("<f8"))

    validate_timestamps("radar", radar_ts_full)
    validate_timestamps("camera", camera_ts)
    validate_timestamps("lidar", lidar_ts_full)

    if len(radar_valid_full) != len(radar_ts_full):
        raise RuntimeError(
            f"Radar valid count {len(radar_valid_full)} does not match "
            f"timestamp count {len(radar_ts_full)}"
        )

    common_start = max(
        float(radar_ts_full[0]),
        float(camera_ts[0]),
        float(lidar_ts_full[0]),
    )
    common_end = min(
        float(radar_ts_full[-1]),
        float(camera_ts[-1]),
        float(lidar_ts_full[-1]),
    )

    t0 = common_start + args.margin_sec
    t1 = common_end - args.margin_sec

    print("[INFO] radar interval :", radar_ts_full[0], radar_ts_full[-1])
    print("[INFO] camera interval:", camera_ts[0], camera_ts[-1])
    print("[INFO] lidar interval :", lidar_ts_full[0], lidar_ts_full[-1])
    print("[INFO] common interval:", common_start, common_end)
    print("[INFO] trim interval  :", t0, t1)

    if common_end <= common_start:
        raise RuntimeError("Sensors have no common time interval")
    if t1 <= t0:
        raise RuntimeError(
            "Common interval is shorter than the selected margins"
        )

    keep_radar = np.flatnonzero(
        (radar_ts_full >= t0) & (radar_ts_full <= t1)
    )
    keep_lidar = np.flatnonzero(
        (lidar_ts_full >= t0) & (lidar_ts_full <= t1)
    )

    if len(keep_radar) == 0:
        raise RuntimeError("No radar frames inside common interval")
    if len(keep_lidar) == 0:
        raise RuntimeError("No lidar frames inside common interval")

    # Do not create a misleading output until all basic validation passes.
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    radar = dst / "radar"
    lidar = dst / "lidar"
    sync = dst / "sync"
    aligned = sync / "aligned_frames_undist"

    # 1. Trim radar timestamps and validity.
    radar_ts = radar_ts_full[keep_radar]
    radar_valid = radar_valid_full[keep_radar]
    radar_ts.astype("<f8").tofile(radar / "ts")
    radar_valid.astype("u1").tofile(radar / "valid")

    print("[INFO] full radar frames:", len(radar_ts_full))
    print("[INFO] keep radar frames:", len(radar_ts))
    print("[INFO] trim duration:", t1 - t0)

    # 2. Trim radar IQ.
    src_iq = src_radar / "iq"
    dst_iq = radar / "iq"

    with src_iq.open("rb") as fin, dst_iq.open("wb") as fout:
        for index in keep_radar:
            fin.seek(int(index) * RADAR_FRAME_BYTES)
            buf = fin.read(RADAR_FRAME_BYTES)
            if len(buf) != RADAR_FRAME_BYTES:
                raise RuntimeError(f"Short read at radar frame {index}")
            fout.write(buf)

    # 3. Trim LiDAR consistently.
    if is_raw_mid360(src_lidar):
        lidar_ts = trim_raw_mid360(
            src_lidar=src_lidar,
            dst_lidar=lidar,
            keep_lidar=keep_lidar,
        )
        lidar_mode = "raw_mid360"
    else:
        lidar_ts = trim_pseudo_lidar(
            src_lidar=src_lidar,
            dst_lidar=lidar,
            keep_lidar=keep_lidar,
        )
        lidar_mode = "pseudo_rng"

    # 4. Rebuild the existing camera manifest and aligned images.
    old_manifest_path = src / "sync" / "nn_manifest.csv"
    with old_manifest_path.open("r", newline="", encoding="utf-8") as f:
        old_manifest = list(csv.DictReader(f))

    old_by_radar = {
        int(row["radar_sorted_i"]): row
        for row in old_manifest
    }

    if aligned.exists():
        shutil.rmtree(aligned)
    ensure_dir(aligned)

    new_manifest_path = sync / "nn_manifest.csv"
    triad_manifest_path = sync / "radar_camera_lidar_manifest.csv"

    lidar_nn = nearest_indices(lidar_ts, radar_ts)

    with (
        new_manifest_path.open("w", newline="", encoding="utf-8") as f_cam,
        triad_manifest_path.open("w", newline="", encoding="utf-8") as f_all,
    ):
        camera_writer = csv.writer(f_cam)
        triad_writer = csv.writer(f_all)

        camera_writer.writerow(
            [
                "radar_sorted_i",
                "radar_sec",
                "camera_sorted_j",
                "camera_orig_idx",
                "camera_sec",
                "delta_camera_minus_radar_sec",
                "image_path_undist",
            ]
        )

        triad_writer.writerow(
            [
                "radar_sorted_i",
                "radar_sec",
                "camera_sorted_j",
                "camera_orig_idx",
                "camera_sec",
                "delta_camera_minus_radar_sec",
                "lidar_sorted_k",
                "lidar_sec",
                "delta_lidar_minus_radar_sec",
                "image_path_undist",
            ]
        )

        for new_i, (old_i, rts) in enumerate(zip(keep_radar, radar_ts)):
            old_i = int(old_i)
            if old_i not in old_by_radar:
                raise RuntimeError(
                    f"Radar frame {old_i} is missing from old manifest"
                )

            old_row = old_by_radar[old_i]
            camera_sorted_j = int(old_row["camera_sorted_j"])
            camera_orig_idx = int(old_row["camera_orig_idx"])
            camera_sec = float(old_row["camera_sec"])
            camera_dt = float(
                old_row["delta_camera_minus_radar_sec"]
            )

            old_img = Path(old_row["image_path_undist"])
            if not old_img.is_absolute():
                old_img = src / old_img
            if not old_img.exists():
                raise FileNotFoundError(
                    f"Missing source image: {old_img}"
                )

            new_img = aligned / f"frame_{new_i:06d}.jpg"
            shutil.copy2(old_img, new_img)

            lidar_k = int(lidar_nn[new_i])
            lidar_sec = float(lidar_ts[lidar_k])
            lidar_dt = lidar_sec - float(rts)

            camera_writer.writerow(
                [
                    new_i,
                    float(rts),
                    camera_sorted_j,
                    camera_orig_idx,
                    camera_sec,
                    camera_dt,
                    str(new_img),
                ]
            )

            triad_writer.writerow(
                [
                    new_i,
                    float(rts),
                    camera_sorted_j,
                    camera_orig_idx,
                    camera_sec,
                    camera_dt,
                    lidar_k,
                    lidar_sec,
                    lidar_dt,
                    str(new_img),
                ]
            )

    # 5. Rebuild placeholder radar pose.
    n = len(radar_ts)
    pose_dir = dst / "_radar"
    ensure_dir(pose_dir)
    np.savez_compressed(
        pose_dir / "pose.npz",
        t=radar_ts.astype("<f8"),
        pos=np.zeros((n, 3), dtype="<f8"),
        vel=np.zeros((n, 3), dtype="<f8"),
        acc=np.zeros((n, 3), dtype="<f8"),
        rot=np.repeat(np.eye(3, dtype="<f8")[None, :, :], n, axis=0),
        mask=np.ones((n,), dtype=bool),
        smoothing=np.array(0.0, dtype="<f8"),
        start_threshold=np.array(0.0, dtype="<f8"),
        filter_size=np.array(0, dtype=np.int64),
    )

    print("[INFO] lidar mode:", lidar_mode)
    print("[INFO] triad manifest:", triad_manifest_path)
    print("[OK] trimmed scene saved:", dst)


if __name__ == "__main__":
    main()

