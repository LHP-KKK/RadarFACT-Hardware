#!/usr/bin/env bash
set -eo pipefail

CAMERA_SETUP="${CAMERA_SETUP:-$HOME/ws_hik/install/setup.bash}"
CAMERA_PACKAGE="${CAMERA_PACKAGE:-camera}"
CAMERA_LAUNCH="${CAMERA_LAUNCH:-camera.launch.py}"

source /opt/ros/humble/setup.bash
source "$CAMERA_SETUP"

ros2 launch "$CAMERA_PACKAGE" "$CAMERA_LAUNCH"
