#!/usr/bin/env python3
"""Export Radar + Camera + Livox Mid-360 rosbag data to an IQ1M-like scene.

Final LiDAR architecture:
    - Acquisition: rosbag2 records /livox/lidar at the full publish rate.
    - Export: this script deserializes Livox CustomMsg offline.
    - Output: variable-length flat raw Mid-360 files.

LiDAR timestamps:
    ts         = rosbag2 record timestamp in Jetson host Unix time
    sensor_ts  = Livox CustomMsg header.stamp
    timebase   = Livox CustomMsg timebase
    offset_time= per-point offset relative to timebase

No custom real-time Python LiDAR recorder is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


LIDAR_BUFFER_BYTES = 16 * 1024 * 1024


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, ensure_ascii=False)


def guess_storage_id(bag_dir: Path) -> str:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        return "sqlite3"

    metadata = yaml.safe_load(
        metadata_path.read_text(encoding="utf-8")
    )
    return metadata.get(
        "rosbag2_bagfile_information",
        {},
    ).get("storage_identifier", "sqlite3")


def open_reader(
    bag_dir: Path,
) -> tuple[SequentialReader, dict[str, str]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(
            uri=str(bag_dir),
            storage_id=guess_storage_id(bag_dir),
        ),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    topic_types = {
        item.name: item.type
        for item in reader.get_all_topics_and_types()
    }
    return reader, topic_types


def stamp_to_sec(
    msg: Any,
    bag_time_ns: int,
    max_skew_sec: float = 10.0,
) -> float:
    """Use a valid host-domain header stamp; otherwise use rosbag time."""
    bag_sec = float(bag_time_ns) * 1e-9

    if hasattr(msg, "header"):
        stamp = msg.header.stamp
        header_sec = (
            float(getattr(stamp, "sec", 0))
            + float(getattr(stamp, "nanosec", 0)) * 1e-9
        )

        if (
            header_sec > 0.0
            and abs(header_sec - bag_sec) <= max_skew_sec
        ):
            return header_sec

    return bag_sec


def header_stamp_to_sec(msg: Any) -> float:
    if not hasattr(msg, "header"):
        return 0.0

    stamp = msg.header.stamp
    return (
        float(getattr(stamp, "sec", 0))
        + float(getattr(stamp, "nanosec", 0)) * 1e-9
    )


def nearest_indices(
    reference_ts: np.ndarray,
    query_ts: np.ndarray,
) -> np.ndarray:
    if len(reference_ts) == 0:
        raise RuntimeError("Reference timestamp array is empty")

    index = np.searchsorted(reference_ts, query_ts)
    index = np.clip(index, 0, len(reference_ts) - 1)

    index0 = np.clip(index - 1, 0, len(reference_ts) - 1)
    index1 = index

    distance0 = np.abs(reference_ts[index0] - query_ts)
    distance1 = np.abs(reference_ts[index1] - query_ts)

    return np.where(
        distance0 <= distance1,
        index0,
        index1,
    )


def write_scalar(
    file_obj: Any,
    value: int | float,
    dtype: str,
) -> None:
    file_obj.write(
        np.asarray(value, dtype=np.dtype(dtype)).tobytes()
    )


class BagRawMid360Writer:
    """Offline Livox CustomMsg -> flat raw Mid-360 writer."""

    def __init__(self, output: Path) -> None:
        self.output = output
        ensure_dir(output)

        self.files = {
            "ts": open(
                output / "ts",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "sensor_ts": open(
                output / "sensor_ts",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "timebase": open(
                output / "timebase",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "lidar_id": open(
                output / "lidar_id",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "point_start": open(
                output / "point_start",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "point_count": open(
                output / "point_count",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "xyz": open(
                output / "xyz",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "offset_time": open(
                output / "offset_time",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "reflectivity": open(
                output / "reflectivity",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "tag": open(
                output / "tag",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
            "line": open(
                output / "line",
                "wb",
                buffering=LIDAR_BUFFER_BYTES,
            ),
        }

        self.frame_count = 0
        self.point_cursor = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None

    def write_message(
        self,
        msg: Any,
        bag_time_ns: int,
    ) -> None:
        host_ts = float(bag_time_ns) * 1e-9
        sensor_ts = header_stamp_to_sec(msg)

        points = msg.points
        point_count = len(points)

        declared_count = int(
            getattr(msg, "point_num", point_count)
        )
        if declared_count != point_count:
            raise RuntimeError(
                "Livox point count mismatch at frame "
                f"{self.frame_count}: "
                f"point_num={declared_count}, "
                f"len(points)={point_count}"
            )

        xyz = np.empty((point_count, 3), dtype="<f4")
        offset_time = np.empty(point_count, dtype="<u4")
        reflectivity = np.empty(point_count, dtype="u1")
        tag = np.empty(point_count, dtype="u1")
        line = np.empty(point_count, dtype="u1")

        # This loop is deliberately offline. It cannot reduce acquisition FPS.
        for index, point in enumerate(points):
            xyz[index, 0] = point.x
            xyz[index, 1] = point.y
            xyz[index, 2] = point.z
            offset_time[index] = point.offset_time
            reflectivity[index] = point.reflectivity
            tag[index] = point.tag
            line[index] = point.line

        write_scalar(self.files["ts"], host_ts, "<f8")
        write_scalar(self.files["sensor_ts"], sensor_ts, "<f8")
        write_scalar(
            self.files["timebase"],
            int(msg.timebase),
            "<u8",
        )
        write_scalar(
            self.files["lidar_id"],
            int(msg.lidar_id),
            "u1",
        )
        write_scalar(
            self.files["point_start"],
            self.point_cursor,
            "<u8",
        )
        write_scalar(
            self.files["point_count"],
            point_count,
            "<u4",
        )

        self.files["xyz"].write(xyz.tobytes(order="C"))
        self.files["offset_time"].write(
            offset_time.tobytes(order="C")
        )
        self.files["reflectivity"].write(
            reflectivity.tobytes(order="C")
        )
        self.files["tag"].write(tag.tobytes(order="C"))
        self.files["line"].write(line.tobytes(order="C"))

        if self.first_ts is None:
            self.first_ts = host_ts
        self.last_ts = host_ts

        self.frame_count += 1
        self.point_cursor += point_count

        if (
            self.frame_count == 1
            or self.frame_count % 100 == 0
        ):
            print(
                "[LIDAR] frames=%d points=%d"
                % (
                    self.frame_count,
                    self.point_cursor,
                ),
                flush=True,
            )

    def close(self) -> None:
        for file_obj in self.files.values():
            file_obj.flush()
            file_obj.close()

        schema = {
            "model": "Livox Mid-360",
            "format": "livox_custom_flat_raw_v2",
            "compression": "none",
            "timestamp_semantics": {
                "ts": (
                    "rosbag2 record timestamp in Jetson host "
                    "Unix time"
                ),
                "sensor_ts": "Livox CustomMsg header.stamp",
                "timebase": "Livox CustomMsg timebase",
                "offset_time": (
                    "Per-point offset relative to Livox timebase"
                ),
            },
            "frame_channels": {
                "ts": "<f8",
                "sensor_ts": "<f8",
                "timebase": "<u8",
                "lidar_id": "u1",
                "point_start": "<u8",
                "point_count": "<u4",
            },
            "point_channels": {
                "xyz": ["<f4", 3],
                "offset_time": "<u4",
                "reflectivity": "u1",
                "tag": "u1",
                "line": "u1",
            },
        }

        meta = {
            "ts": {
                "format": "raw",
                "type": "f8",
                "shape": [],
                "desc": "rosbag2 host Unix timestamp",
            },
            "sensor_ts": {
                "format": "raw",
                "type": "f8",
                "shape": [],
                "desc": "Livox header/device timestamp",
            },
            "timebase": {
                "format": "raw",
                "type": "u8",
                "shape": [],
            },
            "lidar_id": {
                "format": "raw",
                "type": "u1",
                "shape": [],
            },
            "point_start": {
                "format": "raw",
                "type": "u8",
                "shape": [],
            },
            "point_count": {
                "format": "raw",
                "type": "u4",
                "shape": [],
            },
            "xyz": {
                "format": "raw",
                "type": "f4",
                "shape": [3],
            },
            "offset_time": {
                "format": "raw",
                "type": "u4",
                "shape": [],
            },
            "reflectivity": {
                "format": "raw",
                "type": "u1",
                "shape": [],
            },
            "tag": {
                "format": "raw",
                "type": "u1",
                "shape": [],
            },
            "line": {
                "format": "raw",
                "type": "u1",
                "shape": [],
            },
        }

        lidar_json = {
            "sensor": "Livox Mid360",
            "representation": "livox_custom_flat_raw_v2",
            "unit_xyz": "m",
            "compression": "none",
            "source": "rosbag2 /livox/lidar CustomMsg",
            "note": (
                "Variable-length raw point frames indexed by "
                "point_start/point_count."
            ),
        }

        write_json(self.output / "schema.json", schema)
        write_json(self.output / "meta.json", meta)
        write_json(self.output / "lidar.json", lidar_json)


def validate_scene_inputs(
    bag_dir: Path,
    radar_trace: Path,
    out: Path,
) -> None:
    if not bag_dir.is_dir():
        raise FileNotFoundError(
            f"Rosbag directory not found: {bag_dir}"
        )

    radar_dir = radar_trace / "radar"
    if not radar_dir.is_dir():
        raise FileNotFoundError(
            f"Radar directory not found: {radar_dir}"
        )

    if out.exists():
        raise FileExistsError(
            f"Output already exists: {out}. "
            "Remove it before exporting."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--radar-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--image-topic", default="/image_raw")
    parser.add_argument("--lidar-topic", default="/livox/lidar")
    parser.add_argument("--scene", default="indoor_forward")
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=1024)
    args = parser.parse_args()

    bag_dir = Path(args.bag).expanduser().resolve()
    radar_trace = Path(args.radar_trace).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    validate_scene_inputs(
        bag_dir=bag_dir,
        radar_trace=radar_trace,
        out=out,
    )

    radar_out = out / "radar"
    camera_out = out / "camera"
    camera_proc = out / "_camera"
    lidar_out = out / "lidar"
    radar_proc = out / "_radar"
    sync_out = out / "sync"
    aligned_dir = sync_out / "aligned_frames_undist"
    calib_dir = out / "_calib"

    for path in (
        radar_out,
        camera_out,
        camera_proc,
        lidar_out,
        radar_proc,
        sync_out,
        aligned_dir,
        calib_dir,
    ):
        ensure_dir(path)

    source_radar = radar_trace / "radar"
    for name in (
        "iq",
        "ts",
        "valid",
        "meta.json",
        "radar.json",
    ):
        source = source_radar / name
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing radar file: {source}"
            )
        shutil.copy2(source, radar_out / name)

    radar_ts = np.fromfile(
        radar_out / "ts",
        dtype="<f8",
    )

    reader, topic_types = open_reader(bag_dir)

    if args.image_topic not in topic_types:
        raise RuntimeError(
            f"Image topic not found: {args.image_topic}"
        )
    if args.lidar_topic not in topic_types:
        raise RuntimeError(
            f"LiDAR topic not found: {args.lidar_topic}"
        )

    image_message_type = get_message(
        topic_types[args.image_topic]
    )
    lidar_message_type = get_message(
        topic_types[args.lidar_topic]
    )

    bridge = CvBridge()
    lidar_writer = BagRawMid360Writer(lidar_out)

    camera_ts: list[float] = []
    camera_paths: list[Path] = []
    camera_index = 0
    video_writer = None
    video_size: tuple[int, int] | None = None

    segment_writer = lzma.open(
        camera_proc / "segment",
        "wb",
        format=lzma.FORMAT_XZ,
    )

    try:
        while reader.has_next():
            topic, data, bag_time_ns = reader.read_next()

            if topic == args.image_topic:
                msg = deserialize_message(
                    data,
                    image_message_type,
                )
                timestamp = stamp_to_sec(
                    msg,
                    bag_time_ns,
                )
                frame = bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )

                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_size = (width, height)

                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    video_writer = cv2.VideoWriter(
                        str(camera_proc / "video.avi"),
                        fourcc,
                        args.camera_fps,
                        video_size,
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(
                            "Failed to open MJPEG video writer"
                        )

                video_writer.write(frame)

                temporary_path = (
                    aligned_dir
                    / f"_camera_orig_{camera_index:06d}.jpg"
                )

                written = cv2.imwrite(
                    str(temporary_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not written:
                    raise RuntimeError(
                        f"Failed to write image: {temporary_path}"
                    )

                camera_ts.append(timestamp)
                camera_paths.append(temporary_path)
                camera_index += 1

                segment_writer.write(
                    np.zeros(
                        (160, 160),
                        dtype=np.uint8,
                    ).tobytes()
                )

            elif topic == args.lidar_topic:
                msg = deserialize_message(
                    data,
                    lidar_message_type,
                )
                lidar_writer.write_message(
                    msg=msg,
                    bag_time_ns=bag_time_ns,
                )

    finally:
        if video_writer is not None:
            video_writer.release()

        segment_writer.close()
        lidar_writer.close()

    camera_ts_array = np.asarray(
        camera_ts,
        dtype="<f8",
    )

    if len(camera_ts_array) == 0:
        raise RuntimeError("No camera frames were exported")
    if lidar_writer.frame_count == 0:
        raise RuntimeError("No LiDAR frames were exported")
    if len(radar_ts) == 0:
        raise RuntimeError("Radar timestamp stream is empty")

    camera_ts_array.tofile(camera_out / "ts")
    camera_ts_array.tofile(camera_proc / "ts")

    camera_nearest = nearest_indices(
        camera_ts_array,
        radar_ts,
    )

    manifest_path = sync_out / "nn_manifest.csv"
    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
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

        for radar_index, radar_sec in enumerate(radar_ts):
            camera_index = int(
                camera_nearest[radar_index]
            )
            destination_image = (
                aligned_dir
                / f"frame_{radar_index:06d}.jpg"
            )
            shutil.copy2(
                camera_paths[camera_index],
                destination_image,
            )

            writer.writerow(
                [
                    radar_index,
                    float(radar_sec),
                    camera_index,
                    camera_index,
                    float(camera_ts_array[camera_index]),
                    float(
                        camera_ts_array[camera_index]
                        - radar_sec
                    ),
                    str(destination_image),
                ]
            )

    for path in aligned_dir.glob("_camera_orig_*.jpg"):
        path.unlink()

    actual_width = (
        video_size[0]
        if video_size is not None
        else args.camera_width
    )
    actual_height = (
        video_size[1]
        if video_size is not None
        else args.camera_height
    )

    write_json(
        camera_out / "meta.json",
        {
            "ts": {
                "format": "raw",
                "type": "f8",
                "shape": [],
                "desc": "camera timestamps",
            }
        },
    )

    write_json(
        camera_proc / "meta.json",
        {
            "ts": {
                "format": "raw",
                "type": "f8",
                "shape": [],
                "desc": "camera timestamps",
            },
            "video.avi": {
                "format": "mjpg",
                "type": "u1",
                "shape": [
                    actual_height,
                    actual_width,
                    3,
                ],
            },
            "segment": {
                "format": "lzmaf",
                "type": "u1",
                "shape": [160, 160],
                "desc": "placeholder semantic mask",
            },
        },
    )

    radar_frame_count = len(radar_ts)
    np.savez_compressed(
        radar_proc / "pose.npz",
        t=radar_ts.astype("<f8"),
        pos=np.zeros(
            (radar_frame_count, 3),
            dtype="<f8",
        ),
        vel=np.zeros(
            (radar_frame_count, 3),
            dtype="<f8",
        ),
        acc=np.zeros(
            (radar_frame_count, 3),
            dtype="<f8",
        ),
        rot=np.repeat(
            np.eye(3, dtype="<f8")[None, :, :],
            radar_frame_count,
            axis=0,
        ),
        mask=np.ones(
            (radar_frame_count,),
            dtype=bool,
        ),
        smoothing=np.array(0.0, dtype="<f8"),
        start_threshold=np.array(0.0, dtype="<f8"),
        filter_size=np.array(0, dtype=np.int64),
    )

    with (out / "config.yaml").open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        yaml.safe_dump(
            {
                "scene": args.scene,
                "type": "iq1m_minimal_demo",
                "camera": {
                    "type": "camera",
                    "args": {
                        "topic": args.image_topic,
                        "fps": args.camera_fps,
                    },
                },
                "lidar": {
                    "type": "livox_mid360",
                    "args": {
                        "topic": args.lidar_topic,
                        "export": (
                            "livox_custom_flat_raw_v2"
                        ),
                        "source": "rosbag2",
                    },
                },
                "radar": {
                    "type": "radar",
                    "source_trace": str(radar_trace),
                },
                "notes": [
                    "No hardware trigger synchronization.",
                    (
                        "LiDAR ts uses rosbag2 record time in "
                        "the Jetson host clock."
                    ),
                    (
                        "Livox sensor_ts/timebase/offset_time "
                        "are preserved for later deskewing."
                    ),
                    (
                        "Radar-camera alignment uses nearest "
                        "timestamps."
                    ),
                    (
                        "Segment and radar pose are placeholders "
                        "for pipeline testing."
                    ),
                ],
            },
            file_obj,
            sort_keys=False,
            allow_unicode=True,
        )

    lidar_duration = (
        0.0
        if lidar_writer.first_ts is None
        or lidar_writer.last_ts is None
        else lidar_writer.last_ts - lidar_writer.first_ts
    )
    lidar_fps = (
        0.0
        if lidar_writer.frame_count < 2
        or lidar_duration <= 0.0
        else (
            (lidar_writer.frame_count - 1)
            / lidar_duration
        )
    )

    print("[OK] export finished")
    print("out:", out)
    print("camera frames:", len(camera_ts_array))
    print("lidar frames:", lidar_writer.frame_count)
    print("lidar duration:", lidar_duration)
    print("lidar average fps:", lidar_fps)
    print("lidar points:", lidar_writer.point_cursor)
    print("radar frames:", len(radar_ts))
    print("manifest rows:", len(radar_ts))
    print(
        "lidar representation:",
        "livox_custom_flat_raw_v2",
    )


if __name__ == "__main__":
    main()
