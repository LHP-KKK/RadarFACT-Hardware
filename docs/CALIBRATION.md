# Calibration guide

## Camera intrinsics

Use the ChArUco utilities in `software/calibration/` and capture images across
the full field of view, with different distances and board orientations.
Record the exact dictionary, board dimensions, square size, marker size,
resolution, ROI, lens, focus, and aperture.

The included `camera_intrinsics_charuco.yaml` is an example for one physical
configuration. Recalibrate after any optical or imaging change.

## Cross-sensor extrinsics

Store each transform with an explicit source, target, direction, and unit:

```json
{
  "source_sensor": "livox_mid360",
  "target_sensor": "camera",
  "transform_direction": "lidar_to_camera",
  "translation_unit": "meters",
  "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "t": [0, 0, 0]
}
```

Never infer whether a matrix is `T_camera_lidar` or `T_lidar_camera` from its
filename alone. Validate it by projecting points onto strong edges such as
door frames and planar boundaries.

## Time-offset diagnosis

- Fixed spatial error while stationary usually suggests intrinsics/extrinsics.
- Error that grows with motion usually suggests timestamp offset or deskew.
- A delay that changes over time suggests queueing, clock drift, or dropped
  packets and cannot be corrected by one constant offset.

