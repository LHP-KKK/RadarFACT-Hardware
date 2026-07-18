#!/usr/bin/env bash
set -eo pipefail

LIVOX_SETUP="${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}"
LIVOX_LAUNCH="${LIVOX_LAUNCH:-msg_MID360_launch.py}"

source /opt/ros/humble/setup.bash
source "$LIVOX_SETUP"

# Configure the host and sensor IP addresses before launching.
ros2 launch livox_ros_driver2 "$LIVOX_LAUNCH"
