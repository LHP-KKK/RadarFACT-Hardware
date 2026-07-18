#!/usr/bin/env bash
set -Ee -o pipefail

# All site-specific paths and topics can be overridden by environment variables.
SESSION_NAME="${1:-indoor_forward_01}"
IMAGE_TOPIC="${IMAGE_TOPIC:-/image_raw}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"

DATA_ROOT="${DATA_ROOT:-/mnt/sensor_data/demo_traces}"
RADAR_ROOT="${RADAR_ROOT:-$DATA_ROOT/raw_radar}"
ROSBAG_ROOT="${ROSBAG_ROOT:-$DATA_ROOT/raw_rosbags}"
SESSION_BAG="$ROSBAG_ROOT/$SESSION_NAME"

RED_ROVER_ROOT="${RED_ROVER_ROOT:-$HOME/red-rover/collect}"
RADAR_CONFIG="${RADAR_CONFIG:-config/custom/grt-i-demo-radar.yaml}"
LIVOX_SETUP="${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}"
CAMERA_SETUP="${CAMERA_SETUP:-$HOME/ws_hik/install/setup.bash}"

mkdir -p "$RADAR_ROOT" "$ROSBAG_ROOT"

if [[ -e "$SESSION_BAG" ]]; then
    echo "[ERROR] rosbag output already exists: $SESSION_BAG"
    exit 1
fi

for required in /opt/ros/humble/setup.bash "$LIVOX_SETUP" "$CAMERA_SETUP"; do
    if [[ ! -f "$required" ]]; then
        echo "[ERROR] setup file not found: $required"
        exit 1
    fi
done

if [[ ! -d "$RED_ROVER_ROOT" ]]; then
    echo "[ERROR] Red Rover directory not found: $RED_ROVER_ROOT"
    exit 1
fi

source /opt/ros/humble/setup.bash
source "$LIVOX_SETUP"
source "$CAMERA_SETUP"

BAG_PID=""
RADAR_STARTED=0
CLEANUP_DONE=0
TRACE_MARKER="$RADAR_ROOT/.trace_marker_${SESSION_NAME}_$$"
touch "$TRACE_MARKER"

wait_for_topic() {
    local topic="$1"
    local timeout_sec="${2:-20}"
    local start_time
    start_time="$(date +%s)"
    echo "[INFO] waiting for topic: $topic"
    while ! ros2 topic list | grep -Fxq "$topic"; do
        if (( "$(date +%s)" - start_time >= timeout_sec )); then
            echo "[ERROR] topic not found within ${timeout_sec}s: $topic"
            return 1
        fi
        sleep 1
    done
}

cleanup() {
    local exit_status=$?
    if (( CLEANUP_DONE )); then return; fi
    CLEANUP_DONE=1
    set +e

    if (( RADAR_STARTED )); then
        cd "$RED_ROVER_ROOT" || true
        ./.venv/bin/python cli.py stop --config "$RADAR_CONFIG" || true
        RADAR_STARTED=0
    fi

    sleep 2
    if [[ "${BAG_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$BAG_PID" 2>/dev/null; then
        kill -INT "$BAG_PID" 2>/dev/null || true
        wait "$BAG_PID" 2>/dev/null || true
    fi

    date --iso-8601=ns | tee "$ROSBAG_ROOT/${SESSION_NAME}_host_stop.txt"
    rm -f "$TRACE_MARKER"
    echo "[INFO] cleanup finished (original status: $exit_status)"
}

trap 'exit 130' INT TERM
trap cleanup EXIT

wait_for_topic "$IMAGE_TOPIC" 20
wait_for_topic "$LIDAR_TOPIC" 20
wait_for_topic "$IMU_TOPIC" 20

date --iso-8601=ns | tee "$ROSBAG_ROOT/${SESSION_NAME}_host_start.txt"

ros2 bag record \
    --storage sqlite3 \
    --max-cache-size 1073741824 \
    -o "$SESSION_BAG" \
    "$IMAGE_TOPIC" \
    "$LIDAR_TOPIC" \
    "$IMU_TOPIC" &
BAG_PID=$!

sleep 2
if ! kill -0 "$BAG_PID" 2>/dev/null; then
    echo "[ERROR] rosbag exited during startup"
    wait "$BAG_PID" || true
    exit 1
fi

cd "$RED_ROVER_ROOT"
./.venv/bin/python cli.py start --config "$RADAR_CONFIG" --path "$RADAR_ROOT"
RADAR_STARTED=1

LATEST_TRACE=""
for _ in $(seq 1 20); do
    LATEST_TRACE="$(
        find "$RADAR_ROOT" -mindepth 1 -maxdepth 1 -type d -newer "$TRACE_MARKER" \
            -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
    )"
    [[ -n "$LATEST_TRACE" ]] && break
    sleep 0.5
done

if [[ -z "$LATEST_TRACE" ]]; then
    echo "[ERROR] failed to locate the new radar trace directory"
    exit 1
fi

printf '%s\n' "$LATEST_TRACE" | tee "$ROSBAG_ROOT/${SESSION_NAME}_radar_trace.txt"
rm -f "$TRACE_MARKER"

echo "[INFO] recording session: $SESSION_NAME"
echo "[INFO] camera: $IMAGE_TOPIC"
echo "[INFO] lidar : $LIDAR_TOPIC"
echo "[INFO] imu   : $IMU_TOPIC"
echo "[INFO] radar : $LATEST_TRACE"
echo "[INFO] press Ctrl+C to stop"
wait "$BAG_PID"

