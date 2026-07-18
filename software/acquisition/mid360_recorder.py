#!/usr/bin/env python3
"""Low-overhead Livox Mid-360 recorder.

Acquisition path:
    ROS CustomMsg callback
      -> capture host receive timestamp
      -> serialize the whole ROS message to CDR bytes
      -> enqueue one compact Python tuple
      -> background writer appends bytes and frame metadata

After acquisition stops, the script converts the staged CDR packets offline
into the existing flat Mid-360 layout:

    ts, sensor_ts, timebase, lidar_id,
    point_start, point_count,
    xyz, offset_time, reflectivity, tag, line

This avoids the previous real-time Python loop over ~20k points per frame,
which was the main reason the recorder fell to roughly 4 Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message, serialize_message


_STOP = object()


def write_scalar(file_obj, value: int | float, dtype: str) -> None:
    file_obj.write(np.asarray(value, dtype=np.dtype(dtype)).tobytes())


class CdrStageWriter:
    """Append serialized CustomMsg packets with minimal callback overhead."""

    def __init__(
        self,
        output: Path,
        queue_size: int,
        fsync_interval_sec: float,
    ) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)

        self.stage = self.output / "_cdr_stage"
        self.stage.mkdir(parents=True, exist_ok=False)

        buffer_size = 16 * 1024 * 1024

        self.files = {
            "packet_data": open(
                self.stage / "packet_data",
                "wb",
                buffering=buffer_size,
            ),
            "packet_offset": open(
                self.stage / "packet_offset",
                "wb",
                buffering=buffer_size,
            ),
            "packet_size": open(
                self.stage / "packet_size",
                "wb",
                buffering=buffer_size,
            ),
            "ts": open(
                self.stage / "ts",
                "wb",
                buffering=buffer_size,
            ),
            "sensor_ts": open(
                self.stage / "sensor_ts",
                "wb",
                buffering=buffer_size,
            ),
            "timebase": open(
                self.stage / "timebase",
                "wb",
                buffering=buffer_size,
            ),
            "lidar_id": open(
                self.stage / "lidar_id",
                "wb",
                buffering=buffer_size,
            ),
            "point_count": open(
                self.stage / "point_count",
                "wb",
                buffering=buffer_size,
            ),
        }

        self.message_queue: queue.Queue[Any] = queue.Queue(
            maxsize=queue_size
        )

        self.accepted_frames = 0
        self.written_frames = 0
        self.dropped_frames = 0
        self.packet_cursor = 0
        self.error: BaseException | None = None

        self.fsync_interval_sec = fsync_interval_sec
        self.last_fsync_monotonic = time.monotonic()

        self.worker = threading.Thread(
            target=self._writer_loop,
            name="mid360_cdr_writer",
            daemon=False,
        )
        self.worker.start()

    def submit(
        self,
        host_receive_ts: float,
        sensor_ts: float,
        timebase: int,
        lidar_id: int,
        point_count: int,
        cdr_bytes: bytes,
    ) -> bool:
        item = (
            float(host_receive_ts),
            float(sensor_ts),
            int(timebase),
            int(lidar_id),
            int(point_count),
            cdr_bytes,
        )

        try:
            self.message_queue.put_nowait(item)
        except queue.Full:
            self.dropped_frames += 1
            return False

        self.accepted_frames += 1
        return True

    def _flush_and_maybe_fsync(self, force: bool = False) -> None:
        now = time.monotonic()

        if not force and (
            now - self.last_fsync_monotonic
            < self.fsync_interval_sec
        ):
            return

        for file_obj in self.files.values():
            file_obj.flush()

        # Do not fsync every frame. A coarse interval avoids long stalls.
        for file_obj in self.files.values():
            os.fsync(file_obj.fileno())

        self.last_fsync_monotonic = now

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self.message_queue.get()

                if item is _STOP:
                    break

                (
                    host_receive_ts,
                    sensor_ts,
                    timebase,
                    lidar_id,
                    point_count,
                    cdr_bytes,
                ) = item

                packet_size = len(cdr_bytes)

                write_scalar(
                    self.files["packet_offset"],
                    self.packet_cursor,
                    "<u8",
                )
                write_scalar(
                    self.files["packet_size"],
                    packet_size,
                    "<u4",
                )
                write_scalar(
                    self.files["ts"],
                    host_receive_ts,
                    "<f8",
                )
                write_scalar(
                    self.files["sensor_ts"],
                    sensor_ts,
                    "<f8",
                )
                write_scalar(
                    self.files["timebase"],
                    timebase,
                    "<u8",
                )
                write_scalar(
                    self.files["lidar_id"],
                    lidar_id,
                    "u1",
                )
                write_scalar(
                    self.files["point_count"],
                    point_count,
                    "<u4",
                )

                self.files["packet_data"].write(cdr_bytes)

                self.packet_cursor += packet_size
                self.written_frames += 1

                self._flush_and_maybe_fsync(force=False)

        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        self.message_queue.put(_STOP)
        self.worker.join()

        self._flush_and_maybe_fsync(force=True)

        for file_obj in self.files.values():
            file_obj.close()

        if self.error is not None:
            raise RuntimeError(
                "Mid-360 CDR writer thread failed"
            ) from self.error


def convert_cdr_stage_to_flat(
    output: Path,
    keep_cdr: bool,
) -> tuple[int, int]:
    """Convert staged CDR packets to the established flat raw layout."""

    stage = output / "_cdr_stage"

    offsets = np.fromfile(
        stage / "packet_offset",
        dtype="<u8",
    )
    sizes = np.fromfile(
        stage / "packet_size",
        dtype="<u4",
    )
    host_ts = np.fromfile(
        stage / "ts",
        dtype="<f8",
    )
    sensor_ts = np.fromfile(
        stage / "sensor_ts",
        dtype="<f8",
    )
    timebase = np.fromfile(
        stage / "timebase",
        dtype="<u8",
    )
    lidar_id = np.fromfile(
        stage / "lidar_id",
        dtype="u1",
    )
    staged_counts = np.fromfile(
        stage / "point_count",
        dtype="<u4",
    )

    n = len(offsets)

    for name, arr in (
        ("packet_size", sizes),
        ("ts", host_ts),
        ("sensor_ts", sensor_ts),
        ("timebase", timebase),
        ("lidar_id", lidar_id),
        ("point_count", staged_counts),
    ):
        if len(arr) != n:
            raise RuntimeError(
                f"Staged field {name} has {len(arr)} entries; "
                f"expected {n}"
            )

    buffer_size = 16 * 1024 * 1024

    files = {
        "point_start": open(
            output / "point_start",
            "wb",
            buffering=buffer_size,
        ),
        "point_count": open(
            output / "point_count",
            "wb",
            buffering=buffer_size,
        ),
        "xyz": open(
            output / "xyz",
            "wb",
            buffering=buffer_size,
        ),
        "offset_time": open(
            output / "offset_time",
            "wb",
            buffering=buffer_size,
        ),
        "reflectivity": open(
            output / "reflectivity",
            "wb",
            buffering=buffer_size,
        ),
        "tag": open(
            output / "tag",
            "wb",
            buffering=buffer_size,
        ),
        "line": open(
            output / "line",
            "wb",
            buffering=buffer_size,
        ),
    }

    point_cursor = 0

    try:
        with open(stage / "packet_data", "rb") as packet_file:
            for frame_index, (offset, size) in enumerate(
                zip(offsets, sizes)
            ):
                packet_file.seek(int(offset))
                data = packet_file.read(int(size))

                if len(data) != int(size):
                    raise RuntimeError(
                        f"Short CDR read at frame {frame_index}"
                    )

                msg = deserialize_message(data, CustomMsg)
                points = msg.points
                count = len(points)

                if count != int(staged_counts[frame_index]):
                    raise RuntimeError(
                        f"Point count mismatch at frame {frame_index}: "
                        f"staged={int(staged_counts[frame_index])}, "
                        f"decoded={count}"
                    )

                xyz = np.empty((count, 3), dtype="<f4")
                offset_time = np.empty(count, dtype="<u4")
                reflectivity = np.empty(count, dtype="u1")
                tag = np.empty(count, dtype="u1")
                line = np.empty(count, dtype="u1")

                for index, point in enumerate(points):
                    xyz[index, 0] = point.x
                    xyz[index, 1] = point.y
                    xyz[index, 2] = point.z
                    offset_time[index] = point.offset_time
                    reflectivity[index] = point.reflectivity
                    tag[index] = point.tag
                    line[index] = point.line

                write_scalar(
                    files["point_start"],
                    point_cursor,
                    "<u8",
                )
                write_scalar(
                    files["point_count"],
                    count,
                    "<u4",
                )

                files["xyz"].write(xyz.tobytes(order="C"))
                files["offset_time"].write(
                    offset_time.tobytes(order="C")
                )
                files["reflectivity"].write(
                    reflectivity.tobytes(order="C")
                )
                files["tag"].write(tag.tobytes(order="C"))
                files["line"].write(line.tobytes(order="C"))

                point_cursor += count

                if (
                    frame_index == 0
                    or (frame_index + 1) % 100 == 0
                    or frame_index + 1 == n
                ):
                    print(
                        "[CONVERT] frames=%d/%d points=%d"
                        % (
                            frame_index + 1,
                            n,
                            point_cursor,
                        ),
                        flush=True,
                    )

    finally:
        for file_obj in files.values():
            file_obj.flush()
            os.fsync(file_obj.fileno())
            file_obj.close()

    # Publish final per-frame arrays only after conversion succeeds.
    host_ts.astype("<f8").tofile(output / "ts")
    sensor_ts.astype("<f8").tofile(output / "sensor_ts")
    timebase.astype("<u8").tofile(output / "timebase")
    lidar_id.astype("u1").tofile(output / "lidar_id")

    schema = {
        "model": "Livox Mid-360",
        "format": "livox_custom_flat_raw_v2",
        "compression": "none",
        "timestamp_semantics": {
            "ts": "Jetson host receive Unix time captured at ROS callback entry",
            "sensor_ts": "Livox CustomMsg header.stamp",
            "timebase": "Livox CustomMsg timebase",
            "offset_time": "Per-point offset relative to Livox timebase",
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

    with open(
        output / "schema.json",
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(schema, file_obj, indent=2)

    if not keep_cdr:
        shutil.rmtree(stage)

    return n, point_cursor


class Mid360RecorderNode(Node):
    def __init__(
        self,
        topic: str,
        output: Path,
        queue_size: int,
        fsync_interval_sec: float,
    ) -> None:
        super().__init__("mid360_low_overhead_recorder")

        self.writer = CdrStageWriter(
            output=output,
            queue_size=queue_size,
            fsync_interval_sec=fsync_interval_sec,
        )

        # Sensor-data style QoS, but with a larger receive depth so short
        # scheduling stalls do not immediately discard messages.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=max(32, queue_size),
        )

        self.subscription = self.create_subscription(
            CustomMsg,
            topic,
            self.callback,
            qos,
        )

        self.last_log_monotonic = time.monotonic()

        self.get_logger().info(
            f"Recording {topic} to {output}; "
            f"queue_size={queue_size}"
        )

    def callback(self, msg: CustomMsg) -> None:
        # Capture host time immediately on callback entry.
        host_receive_ts = time.time_ns() * 1e-9

        stamp = msg.header.stamp
        sensor_ts = (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

        # rclpy serialization is implemented below Python and avoids a
        # Python-level loop over every point during acquisition.
        cdr_bytes = serialize_message(msg)

        accepted = self.writer.submit(
            host_receive_ts=host_receive_ts,
            sensor_ts=sensor_ts,
            timebase=int(msg.timebase),
            lidar_id=int(msg.lidar_id),
            point_count=len(msg.points),
            cdr_bytes=cdr_bytes,
        )

        if not accepted:
            dropped = self.writer.dropped_frames
            if dropped == 1 or dropped % 10 == 0:
                self.get_logger().error(
                    "Writer queue full: "
                    f"dropped={dropped}, "
                    f"qsize={self.writer.message_queue.qsize()}"
                )

        now = time.monotonic()
        if now - self.last_log_monotonic >= 5.0:
            self.last_log_monotonic = now
            self.get_logger().info(
                "accepted=%d written=%d dropped=%d qsize=%d"
                % (
                    self.writer.accepted_frames,
                    self.writer.written_frames,
                    self.writer.dropped_frames,
                    self.writer.message_queue.qsize(),
                )
            )

    def close_writer(self) -> None:
        self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topic",
        default="/livox/lidar",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--fsync-interval-sec",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--keep-cdr",
        action="store_true",
        help="Keep the staged serialized packets after flat conversion",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()

    if output.exists():
        raise FileExistsError(
            f"Output directory already exists: {output}"
        )

    rclpy.init()

    node = Mid360RecorderNode(
        topic=args.topic,
        output=output,
        queue_size=args.queue_size,
        fsync_interval_sec=args.fsync_interval_sec,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            "Stopping acquisition and draining writer queue..."
        )

        node.close_writer()

        node.get_logger().info(
            "Acquisition closed: accepted=%d written=%d dropped=%d"
            % (
                node.writer.accepted_frames,
                node.writer.written_frames,
                node.writer.dropped_frames,
            )
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    if node.writer.accepted_frames != node.writer.written_frames:
        raise RuntimeError(
            "Accepted/written frame mismatch after queue drain"
        )

    print(
        "[INFO] Converting staged CDR packets to flat Mid-360 files. "
        "This happens after acquisition, so it cannot reduce capture FPS.",
        flush=True,
    )

    frame_count, point_count = convert_cdr_stage_to_flat(
        output=output,
        keep_cdr=args.keep_cdr,
    )

    print(
        "[OK] Mid-360 conversion complete: "
        f"frames={frame_count}, points={point_count}, "
        f"dropped={node.writer.dropped_frames}",
        flush=True,
    )


if __name__ == "__main__":
    main()

