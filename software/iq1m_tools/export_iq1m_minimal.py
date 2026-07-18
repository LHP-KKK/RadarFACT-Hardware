#!/usr/bin/env python3
import os
import csv
import json
import yaml
import lzma
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def guess_storage_id(bag_dir: Path):
    meta = bag_dir / "metadata.yaml"
    if meta.exists():
        y = yaml.safe_load(meta.read_text())
        return y.get("rosbag2_bagfile_information", {}).get("storage_identifier", "sqlite3")
    return "sqlite3"


def open_reader(bag_dir: Path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id=guess_storage_id(bag_dir)),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topic_types


def stamp_to_sec(msg, bag_time_ns, max_skew_sec=10.0):
    """
    Return timestamp in host Unix time domain.

    Prefer msg.header.stamp only when it is close to rosbag record time.
    Some sensors, especially Livox, may publish device/internal timestamps
    that are far from the host Unix epoch. In that case, fall back to bag time.
    """
    bag_sec = bag_time_ns * 1e-9

    if hasattr(msg, "header"):
        sec = getattr(msg.header.stamp, "sec", 0)
        nsec = getattr(msg.header.stamp, "nanosec", 0)
        header_sec = float(sec) + float(nsec) * 1e-9

        if header_sec > 0 and abs(header_sec - bag_sec) <= max_skew_sec:
            return header_sec

    return bag_sec

def nearest_indices(ref_ts, query_ts):
    idx = np.searchsorted(ref_ts, query_ts)
    idx = np.clip(idx, 0, len(ref_ts) - 1)
    idx0 = np.clip(idx - 1, 0, len(ref_ts) - 1)
    idx1 = idx
    d0 = np.abs(ref_ts[idx0] - query_ts)
    d1 = np.abs(ref_ts[idx1] - query_ts)
    return np.where(d0 <= d1, idx0, idx1)


def pc2_to_xyz(msg):
    """
    Robustly convert sensor_msgs/PointCloud2 to plain (N, 3) float32 xyz array.

    In ROS2 Humble, sensor_msgs_py.point_cloud2.read_points() may return either:
    1) a structured numpy array with dtype.names = ('x', 'y', 'z'), or
    2) an iterator/list of tuples.

    This function handles both cases.
    """
    names = [f.name for f in msg.fields]
    for name in ("x", "y", "z"):
        if name not in names:
            raise RuntimeError(
                f"PointCloud2 missing field '{name}', available fields={names}"
            )

    pts = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True
    )

    # Case 1: structured numpy array, common in ROS2 Humble
    if isinstance(pts, np.ndarray):
        if pts.dtype.names is not None:
            arr = np.column_stack(
                (pts["x"], pts["y"], pts["z"])
            ).astype(np.float32, copy=False)
        else:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)

    # Case 2: generator/list of tuples
    else:
        rows = list(pts)
        if len(rows) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.asarray(
            [[r[0], r[1], r[2]] for r in rows],
            dtype=np.float32
        )

    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    arr = arr.reshape(-1, 3)
    arr = arr[np.isfinite(arr).all(axis=1)]
    return arr.astype(np.float32, copy=False)


def livox_custom_to_xyz(msg):
    """
    Convert livox_ros_driver2/msg/CustomMsg to plain (N, 3) float32 xyz array.

    Livox CustomMsg normally has:
      msg.points[i].x
      msg.points[i].y
      msg.points[i].z
      msg.points[i].reflectivity
      msg.points[i].offset_time
    """
    if not hasattr(msg, "points"):
        raise RuntimeError("This message has no 'points' field; not a Livox CustomMsg")

    pts = msg.points
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xyz = np.empty((len(pts), 3), dtype=np.float32)
    for i, pt in enumerate(pts):
        xyz[i, 0] = pt.x
        xyz[i, 1] = pt.y
        xyz[i, 2] = pt.z

    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz.astype(np.float32, copy=False)


def lidar_msg_to_xyz(msg):
    """
    Dispatch LiDAR message to xyz array.

    Supports:
      1. sensor_msgs/msg/PointCloud2
      2. livox_ros_driver2/msg/CustomMsg
    """
    if hasattr(msg, "fields"):
        return pc2_to_xyz(msg)

    if hasattr(msg, "points"):
        return livox_custom_to_xyz(msg)

    raise RuntimeError(
        f"Unsupported lidar message type: {type(msg)}. "
        "Expected PointCloud2-like message with 'fields' or Livox CustomMsg with 'points'."
    )

def points_to_pseudo_rng(points, rows=128, cols=2048, vmin_deg=-45.0, vmax_deg=45.0):
    """
    Livox µãÔÆ -> pseudo range image.
    ×¢Òâ£ºÕâÊÇÎªÁËÏÈÅÜÍ¨ IQ1M-like loader£¬²»µÈ¼ÛÓÚ Ouster OS-0-128 Ô­Ê¼ rng¡£
    """
    rng = np.zeros((rows, cols), dtype=np.uint16)
    if points.shape[0] == 0:
        return rng

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    valid = r > 0.1

    x, y, z, r = x[valid], y[valid], z[valid], r[valid]
    if len(r) == 0:
        return rng

    az = np.arctan2(y, x)
    el = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0)))

    col = ((az + np.pi) / (2 * np.pi) * cols).astype(np.int32)
    row = ((vmax_deg - el) / (vmax_deg - vmin_deg) * rows).astype(np.int32)

    keep = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    row = row[keep]
    col = col[keep]
    r_mm = np.clip(np.round(r[keep] * 1000.0), 0, 65535).astype(np.uint16)

    # Í¬Ò»ÏñËØ±£Áô×î½ü¾àÀë
    for rr, cc, mm in zip(row, col, r_mm):
        old = rng[rr, cc]
        if old == 0 or mm < old:
            rng[rr, cc] = mm

    return rng


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--radar-trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-topic", default="/image_raw")
    ap.add_argument("--lidar-topic", default="/livox/lidar")
    ap.add_argument("--scene", default="indoor_forward_02")
    ap.add_argument("--camera-width", type=int, default=1280)
    ap.add_argument("--camera-height", type=int, default=1024)
    ap.add_argument("--camera-fps", type=float, default=15.0)
    args = ap.parse_args()

    bag_dir = Path(args.bag).expanduser().resolve()
    radar_trace = Path(args.radar_trace).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    radar_out = out / "radar"
    camera_out = out / "camera"
    camera_proc = out / "_camera"
    lidar_out = out / "lidar"
    radar_proc = out / "_radar"
    sync_out = out / "sync"
    aligned_dir = sync_out / "aligned_frames_undist"
    calib_dir = out / "_calib"

    for p in [radar_out, camera_out, camera_proc, lidar_out, radar_proc, sync_out, aligned_dir, calib_dir]:
        ensure_dir(p)

    # 1. copy radar
    src_radar = radar_trace / "radar"
    if not src_radar.exists():
        raise RuntimeError(f"radar dir not found: {src_radar}")
    for name in ["iq", "ts", "valid", "meta.json", "radar.json"]:
        shutil.copy2(src_radar / name, radar_out / name)

    radar_ts = np.fromfile(radar_out / "ts", dtype="<f8")
    radar_valid = np.fromfile(radar_out / "valid", dtype="u1")

    # 2. read rosbag, export camera and lidar
    reader, topic_types = open_reader(bag_dir)
    if args.image_topic not in topic_types:
        raise RuntimeError(f"image topic not found: {args.image_topic}")
    if args.lidar_topic not in topic_types:
        raise RuntimeError(f"lidar topic not found: {args.lidar_topic}")

    ImageMsg = get_message(topic_types[args.image_topic])
    LidarMsg = get_message(topic_types[args.lidar_topic])
    bridge = CvBridge()

    cam_ts = []
    cam_paths = []
    lidar_ts = []

    video_writer = None

    segment_writer = lzma.open(camera_proc / "segment", "wb", format=lzma.FORMAT_XZ)
    rng_writer = lzma.open(lidar_out / "rng", "wb", format=lzma.FORMAT_XZ)

    cam_idx = 0
    lidar_idx = 0

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()

        if topic == args.image_topic:
            msg = deserialize_message(data, ImageMsg)
            ts = stamp_to_sec(msg, bag_time_ns)

            frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if video_writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(
                    str(camera_proc / "video.avi"),
                    fourcc,
                    args.camera_fps,
                    (w, h)
                )

            video_writer.write(frame)

            # ÕâÀïÔÝÊ±Ö±½Ó±£´æÔ­Í¼£»ºóÐø½ÓÈë camera.json ºó¿ÉÌæ»»ÎªÈ¥»û±äÍ¼
            tmp_path = aligned_dir / f"_camera_orig_{cam_idx:06d}.jpg"
            cv2.imwrite(str(tmp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            cam_ts.append(ts)
            cam_paths.append(tmp_path)
            cam_idx += 1

            # placeholder segment£ºÈ« 0
            segment_writer.write(np.zeros((160, 160), dtype=np.uint8).tobytes())

        elif topic == args.lidar_topic:
            msg = deserialize_message(data, LidarMsg)
            ts = stamp_to_sec(msg, bag_time_ns)
            pts = lidar_msg_to_xyz(msg)
            pseudo_rng = points_to_pseudo_rng(pts)
            rng_writer.write(pseudo_rng.tobytes())
            lidar_ts.append(ts)
            lidar_idx += 1

    if video_writer is not None:
        video_writer.release()
    segment_writer.close()
    rng_writer.close()

    cam_ts = np.asarray(cam_ts, dtype="<f8")
    lidar_ts = np.asarray(lidar_ts, dtype="<f8")

    cam_ts.tofile(camera_out / "ts")
    cam_ts.tofile(camera_proc / "ts")
    lidar_ts.tofile(lidar_out / "ts")

    # 3. radar-camera manifest and aligned frames
    cam_nn = nearest_indices(cam_ts, radar_ts)

    manifest_path = sync_out / "nn_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "radar_sorted_i",
            "radar_sec",
            "camera_sorted_j",
            "camera_orig_idx",
            "camera_sec",
            "delta_camera_minus_radar_sec",
            "image_path_undist"
        ])

        for ri, rts in enumerate(radar_ts):
            cj = int(cam_nn[ri])
            src_img = cam_paths[cj]
            dst_img = aligned_dir / f"frame_{ri:06d}.jpg"
            shutil.copy2(src_img, dst_img)

            w.writerow([
                ri,
                float(rts),
                cj,
                cj,
                float(cam_ts[cj]),
                float(cam_ts[cj] - rts),
                str(dst_img)
            ])

    # É¾³ýÁÙÊ±Í¼
    for p in aligned_dir.glob("_camera_orig_*.jpg"):
        p.unlink()

    # 4. meta files
    write_json(camera_out / "meta.json", {
        "ts": {"format": "raw", "type": "f8", "shape": [], "desc": "camera timestamps"}
    })

    write_json(camera_proc / "meta.json", {
        "ts": {"format": "raw", "type": "f8", "shape": [], "desc": "camera timestamps"},
        "video.avi": {"format": "mjpg", "type": "u1", "shape": [args.camera_height, args.camera_width, 3]},
        "segment": {"format": "lzmaf", "type": "u1", "shape": [160, 160], "desc": "placeholder semantic mask"}
    })

    write_json(lidar_out / "meta.json", {
        "ts": {"format": "raw", "type": "f8", "shape": [], "desc": "lidar timestamps"},
        "rng": {"format": "lzmaf", "type": "u2", "shape": [128, 2048], "desc": "pseudo range image from Livox Mid360"}
    })

    write_json(lidar_out / "lidar.json", {
        "sensor": "Livox Mid360",
        "representation": "pseudo_ouster_like_rng",
        "rows": 128,
        "cols": 2048,
        "unit": "mm",
        "note": "Generated by spherical projection from Livox point cloud. Not native Ouster beam-time image."
    })

    # 5. placeholder radar pose
    n = len(radar_ts)
    np.savez_compressed(
        radar_proc / "pose.npz",
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

    # 6. config
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({
            "scene": args.scene,
            "type": "iq1m_minimal_demo",
            "camera": {
                "type": "camera",
                "args": {
                    "topic": args.image_topic,
                    "fps": args.camera_fps
                }
            },
            "lidar": {
                "type": "livox_mid360",
                "args": {
                    "topic": args.lidar_topic,
                    "export": "pseudo_rng_128x2048"
                }
            },
            "radar": {
                "type": "radar",
                "source_trace": str(radar_trace)
            },
            "notes": [
                "No hardware trigger sync.",
                "Radar-camera alignment uses nearest neighbor timestamp matching.",
                "LiDAR rng is pseudo range image from Livox point cloud.",
                "segment and radar pose are placeholders for pipeline smoke test."
            ]
        }, f, sort_keys=False, allow_unicode=True)

    print("[OK] export finished")
    print("out:", out)
    print("camera frames:", len(cam_ts))
    print("lidar frames:", len(lidar_ts))
    print("radar frames:", len(radar_ts))
    print("manifest rows:", len(radar_ts))


if __name__ == "__main__":
    main()
