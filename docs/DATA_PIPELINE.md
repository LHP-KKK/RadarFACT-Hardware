# Data pipeline

## Inputs

- Raw TI radar I/Q trace recorded through DCA1000/Red Rover
- ROS 2 camera images
- Livox Mid-360 `CustomMsg` point clouds
- Livox IMU messages
- Sensor and host timestamps
- Camera intrinsics and cross-sensor extrinsics

## Recommended recording contract

Each session should record:

```text
session_id
hardware revision
radar board, firmware, modulation, antenna configuration
camera serial alias, exposure, gain, pixel format, resolution
LiDAR firmware and network configuration
calibration version and transform direction
sensor timestamp and host receive timestamp
trigger_id and synchronizer tick where available
frame completeness and packet-loss diagnostics
```

Do not publish real device serial numbers or site-specific IP addresses in
shared configuration examples. Use local, untracked configuration files.

## Export stages

1. `export_iq1m_minimal_mid360_bag_raw.py` preserves Mid-360 native point
   coordinates, reflectivity, device timestamps, and per-point offsets.
2. `make_triad_manifest.py` associates radar, camera, and LiDAR timestamps.
3. `undistort_aligned_frames.py` applies the calibrated camera model.
4. `project_lidar_to_camera.py` produces calibrated depth projections.
5. `make_camera_region_cache.py` builds downstream camera-region caches.
6. `trim_iq1m_scene.py` removes unmatched trace tails after inspection.

Nearest-neighbour association is not a substitute for verified hardware
synchronization. Always report the mean, P95, and maximum time residual, and
check for drift over the full session.

## Data policy

Raw recordings are intentionally excluded from Git. Store them on a dedicated
SSD or dataset server and publish only through a dataset release with explicit
consent, licensing, checksums, and metadata.

