#!/usr/bin/env python3
import argparse
import csv
import json
import yaml
from pathlib import Path

import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2


def guess_storage_id(bag_dir: Path):
    meta = bag_dir / "metadata.yaml"
    if not meta.exists():
        return "sqlite3"

    y = yaml.safe_load(meta.read_text())
    return y.get("rosbag2_bagfile_information", {}).get("storage_identifier", "sqlite3")


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
    ·µ»ØÖ÷»ú Unix Ê±¼äÓòÏÂµÄÃë¼¶Ê±¼ä´Á¡£

    ¶Ô Livox CustomMsg À´Ëµ£¬header.stamp ¿ÉÄÜÊÇÉè±¸Ê±¼ä»òÒì³£ epoch£¬
    Òò´ËÖ»ÓÐµ± header.stamp Óë rosbag ½ÓÊÕÊ±¼ä×ã¹»½Ó½üÊ±²ÅÊ¹ÓÃ header£»
    ·ñÔòÍË»Ø bag_time_ns¡£
    """
    bag_sec = bag_time_ns * 1e-9

    if hasattr(msg, "header"):
        sec = getattr(msg.header.stamp, "sec", 0)
        nsec = getattr(msg.header.stamp, "nanosec", 0)
        header_sec = float(sec) + float(nsec) * 1e-9

        if header_sec > 0 and abs(header_sec - bag_sec) <= max_skew_sec:
            return header_sec

    return bag_sec


def pointcloud2_to_arrays(msg):
    """
    sensor_msgs/msg/PointCloud2 -> xyz/intensity arrays
    """
    names = [f.name for f in msg.fields]

    for key in ["x", "y", "z"]:
        if key not in names:
            raise RuntimeError(f"PointCloud2 missing field {key}; fields={names}")

    has_intensity = "intensity" in names
    field_names = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

    pts = point_cloud2.read_points(
        msg,
        field_names=field_names,
        skip_nans=True,
    )

    if isinstance(pts, np.ndarray):
        if pts.dtype.names is not None:
            xyz = np.column_stack([pts["x"], pts["y"], pts["z"]]).astype(np.float32)
            if has_intensity:
                intensity = pts["intensity"].astype(np.float32)
            else:
                intensity = np.zeros((len(xyz),), dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32)
            xyz = arr[:, :3]
            if has_intensity and arr.shape[1] > 3:
                intensity = arr[:, 3].astype(np.float32)
            else:
                intensity = np.zeros((len(xyz),), dtype=np.float32)
    else:
        rows = list(pts)
        if len(rows) == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                {},
            )

        arr = np.asarray(rows, dtype=np.float32)
        xyz = arr[:, :3]
        if has_intensity and arr.shape[1] > 3:
            intensity = arr[:, 3].astype(np.float32)
        else:
            intensity = np.zeros((len(xyz),), dtype=np.float32)

    mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[mask]
    intensity = intensity[mask]

    extra = {
        "message_type": "sensor_msgs/msg/PointCloud2",
        "has_intensity": has_intensity,
        "fields": names,
    }

    return xyz.astype(np.float32), intensity.astype(np.float32), extra


def livox_custom_to_arrays(msg):
    """
    livox_ros_driver2/msg/CustomMsg -> xyz/reflectivity/offset_time/line/tag arrays

    ³£¼û×Ö¶Î£º
      msg.points[i].x
      msg.points[i].y
      msg.points[i].z
      msg.points[i].reflectivity
      msg.points[i].offset_time
      msg.points[i].line
      msg.points[i].tag
    """
    if not hasattr(msg, "points"):
        raise RuntimeError("This message has no points field; not Livox CustomMsg")

    pts = msg.points
    n = len(pts)

    if n == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {
                "offset_time": np.zeros((0,), dtype=np.uint32),
                "line": np.zeros((0,), dtype=np.uint8),
                "tag": np.zeros((0,), dtype=np.uint8),
            },
        )

    xyz = np.empty((n, 3), dtype=np.float32)
    reflectivity = np.empty((n,), dtype=np.float32)
    offset_time = np.zeros((n,), dtype=np.uint32)
    line = np.zeros((n,), dtype=np.uint8)
    tag = np.zeros((n,), dtype=np.uint8)

    for i, p in enumerate(pts):
        xyz[i, 0] = p.x
        xyz[i, 1] = p.y
        xyz[i, 2] = p.z

        reflectivity[i] = float(getattr(p, "reflectivity", 0.0))
        offset_time[i] = int(getattr(p, "offset_time", 0))
        line[i] = int(getattr(p, "line", 0))
        tag[i] = int(getattr(p, "tag", 0))

    mask = np.isfinite(xyz).all(axis=1)

    xyz = xyz[mask]
    reflectivity = reflectivity[mask]
    offset_time = offset_time[mask]
    line = line[mask]
    tag = tag[mask]

    extra = {
        "message_type": "livox_ros_driver2/msg/CustomMsg",
        "offset_time": offset_time,
        "line": line,
        "tag": tag,
    }

    return xyz.astype(np.float32), reflectivity.astype(np.float32), extra


def lidar_msg_to_arrays(msg):
    """
    ×Ô¶¯ÅÐ¶Ï LiDAR ÏûÏ¢ÀàÐÍ£º
      - PointCloud2: msg.fields
      - Livox CustomMsg: msg.points
    """
    if hasattr(msg, "fields"):
        return pointcloud2_to_arrays(msg)

    if hasattr(msg, "points"):
        return livox_custom_to_arrays(msg)

    raise RuntimeError(
        f"Unsupported lidar msg type: {type(msg)}. "
        "Expected PointCloud2-like message with fields or Livox CustomMsg with points."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag directory, e.g. ~/demo_traces/raw_rosbags/indoor_forward_03")
    ap.add_argument("--scene", required=True, help="scene root, e.g. ~/iq1m_demo/indoor_forward_03_trim.fwd")
    ap.add_argument("--topic", default="/livox/lidar")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    bag_dir = Path(args.bag).expanduser().resolve()
    scene = Path(args.scene).expanduser().resolve()

    if not bag_dir.exists():
        raise FileNotFoundError(f"bag not found: {bag_dir}")
    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")

    out_root = scene / "_lidar"
    points_dir = out_root / "points"
    manifest_path = out_root / "points_manifest.csv"
    ts_path = out_root / "ts"

    out_root.mkdir(parents=True, exist_ok=True)

    if points_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(points_dir)

    points_dir.mkdir(parents=True, exist_ok=True)

    reader, topic_types = open_reader(bag_dir)

    if args.topic not in topic_types:
        print("[ERROR] topic not found:", args.topic)
        print("[INFO] available topics:")
        for k, v in topic_types.items():
            print(" ", k, ":", v)
        raise RuntimeError(f"topic not found: {args.topic}")

    msg_type = topic_types[args.topic]
    Msg = get_message(msg_type)

    print("[INFO] bag:", bag_dir)
    print("[INFO] scene:", scene)
    print("[INFO] topic:", args.topic)
    print("[INFO] msg_type:", msg_type)
    print("[INFO] output:", points_dir)

    rows = []
    ts_list = []
    idx = 0

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()

        if topic != args.topic:
            continue

        msg = deserialize_message(data, Msg)
        ts = stamp_to_sec(msg, bag_time_ns)

        xyz, intensity, extra = lidar_msg_to_arrays(msg)

        out_npz = points_dir / f"point_{idx:06d}.npz"

        save_dict = {
            "xyz": xyz.astype(np.float32),
            "intensity": intensity.astype(np.float32),
            "ts": np.array(ts, dtype=np.float64),
            "lidar_sorted_j": np.array(idx, dtype=np.int32),
            "source_topic": np.array(args.topic),
            "source_msg_type": np.array(msg_type),
        }

        if msg_type == "livox_ros_driver2/msg/CustomMsg":
            save_dict["reflectivity"] = intensity.astype(np.float32)
            save_dict["offset_time"] = extra["offset_time"]
            save_dict["line"] = extra["line"]
            save_dict["tag"] = extra["tag"]

        np.savez_compressed(out_npz, **save_dict)

        rows.append([
            idx,
            f"{ts:.9f}",
            str(out_npz),
            int(len(xyz)),
            msg_type,
        ])
        ts_list.append(ts)

        if idx % 50 == 0:
            print(f"[INFO] frame {idx:06d}, points={len(xyz)}, ts={ts:.6f}")

        idx += 1

    np.asarray(ts_list, dtype="<f8").tofile(ts_path)

    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "lidar_sorted_j",
            "lidar_sec",
            "point_path",
            "num_points",
            "msg_type",
        ])
        w.writerows(rows)

    meta = {
        "format": "native_livox_point_cloud_npz",
        "topic": args.topic,
        "message_type": msg_type,
        "num_frames": idx,
        "points_dir": str(points_dir),
        "manifest": str(manifest_path),
        "ts": str(ts_path),
        "npz_fields": {
            "xyz": "float32, shape=[N,3], meters, Livox/LiDAR frame",
            "intensity": "float32, shape=[N], reflectivity/intensity if available",
            "reflectivity": "float32, same as intensity for Livox CustomMsg",
            "offset_time": "uint32, per-point offset time for Livox CustomMsg if available",
            "line": "uint8, Livox line id if available",
            "tag": "uint8, Livox point tag if available",
            "ts": "float64, frame timestamp in host Unix time domain",
            "lidar_sorted_j": "int32, frame index matching points_manifest.csv",
        },
        "note": (
            "This native point cloud export is used for LiDAR-to-camera projection QA and depth generation. "
            "It does not replace lidar/rng pseudo range image used for IQ1M-like compatibility."
        ),
    }

    (out_root / "points_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("[OK] exported native LiDAR points")
    print(" frames:", idx)
    print(" manifest:", manifest_path)
    print(" ts:", ts_path)
    print(" meta:", out_root / "points_meta.json")


if __name__ == "__main__":
    main()
