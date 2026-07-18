# Software tools

## Acquisition

`acquisition/` contains research scripts for starting Livox, Hikrobot, and Red
Rover backends and recording synchronized session assets. Override machine-
specific settings with environment variables; do not edit public scripts to
commit local serial numbers, IP addresses, or credentials.

The main recorder now includes `/livox/imu` by default because point-cloud
deskew and odometry require a real IMU stream.

## Calibration

`calibration/` contains ChArUco generation, capture, detection, calibration,
undistortion validation, and radar range-profile visualization utilities.

Example:

```bash
python software/calibration/undistort_check.py \
  --calibration software/calibration/camera_intrinsics_charuco.yaml \
  --images 'calibration_images/*.jpg'
```

## I/Q-1M-like tools

`iq1m_tools/` contains data export, native Mid-360 point preservation,
timestamp association, calibration projection, cache generation, and trace
trimming tools. Use `--help` on each script for its current interface.

