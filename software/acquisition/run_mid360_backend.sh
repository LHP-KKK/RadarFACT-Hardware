#!/usr/bin/env bash
# ROS setup scripts may not be compatible with nounset.
set -Ee
set -o pipefail

RED_ROVER="${RED_ROVER_ROOT:-$HOME/red-rover/collect}"
CONFIG="${RADAR_CONFIG:-config/custom/grt-i-demo-full.yaml}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LIVOX_SCRIPT="${LIVOX_SCRIPT:-$SCRIPT_DIR/run_livox.sh}"
LIVOX_SETUP="${LIVOX_SETUP:-$HOME/ws_livox/install/setup.bash}"

LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
EXPECTED_TYPE="livox_ros_driver2/msg/CustomMsg"
LIDAR_SOCKET="/tmp/rover/lidar"

LIVOX_PID=""
STARTED_LIVOX=0


cleanup()
{
    status=$?

    trap - EXIT INT TERM

    # Ö»¹Ø±ÕÓÉ±¾½Å±¾Æô¶¯µÄ Livox Çý¶¯¡£
    if [ "$STARTED_LIVOX" -eq 1 ] &&
       [ -n "$LIVOX_PID" ] &&
       kill -0 "$LIVOX_PID" 2>/dev/null; then

        echo
        echo "[INFO] Stopping Livox driver..."

        # Livox Í¨¹ý setsid Æô¶¯£¬Òò´ËÏòÕû¸ö½ø³Ì×é·¢ËÍÐÅºÅ¡£
        kill -INT -- "-$LIVOX_PID" 2>/dev/null || true

        for _ in $(seq 1 50); do
            if ! kill -0 "$LIVOX_PID" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done

        if kill -0 "$LIVOX_PID" 2>/dev/null; then
            kill -TERM -- "-$LIVOX_PID" 2>/dev/null || true
        fi

        wait "$LIVOX_PID" 2>/dev/null || true
    fi

    exit "$status"
}


trap cleanup EXIT INT TERM


# ------------------------------------------------------------
# »·¾³¼°ÎÄ¼þ¼ì²é
# ------------------------------------------------------------

for file in \
    "$RED_ROVER/$CONFIG" \
    "$LIVOX_SCRIPT" \
    "$LIVOX_SETUP"
do
    if [ ! -f "$file" ]; then
        echo "[ERROR] Missing file:"
        echo "        $file"
        exit 1
    fi
done

source /opt/ros/humble/setup.bash
source "$LIVOX_SETUP"


# ------------------------------------------------------------
# ¼ì²éÊÇ·ñÒÑÓÐ Mid-360 ºó¶Ë
# ------------------------------------------------------------

if [ -S "$LIDAR_SOCKET" ] &&
   ss -xl 2>/dev/null | grep -Fq "$LIDAR_SOCKET"; then

    echo "[ERROR] Mid-360 backend is already running:"
    echo "        $LIDAR_SOCKET"
    exit 1
fi

# ÇåÀíºó¶ËÒì³£ÍË³öÁôÏÂµÄÎÞÐ§ socket¡£
rm -f "$LIDAR_SOCKET"


# ------------------------------------------------------------
# ¼ì²é»òÆô¶¯ Livox CustomMsg Çý¶¯
# ------------------------------------------------------------

CURRENT_TYPE="$(
    timeout 2s ros2 topic type "$LIDAR_TOPIC" 2>/dev/null || true
)"

if [ -n "$CURRENT_TYPE" ] &&
   [ "$CURRENT_TYPE" != "$EXPECTED_TYPE" ]; then

    echo "[ERROR] Existing LiDAR topic has the wrong type:"
    echo "        topic   : $LIDAR_TOPIC"
    echo "        current : $CURRENT_TYPE"
    echo "        expected: $EXPECTED_TYPE"
    echo
    echo "Stop the old PointCloud2 Livox driver first."
    exit 1
fi

if [ "$CURRENT_TYPE" = "$EXPECTED_TYPE" ]; then
    echo "[INFO] Reusing existing Livox CustomMsg driver."
else
    echo "[INFO] Starting Livox CustomMsg driver..."

    # ´´½¨¶ÀÁ¢½ø³Ì×é£¬ÍË³ö±¾½Å±¾Ê±¿ÉÍêÕû¹Ø±Õ ros2 launch ¼°Æä×Ó½Úµã¡£
    setsid "$LIVOX_SCRIPT" &

    LIVOX_PID=$!
    STARTED_LIVOX=1

    READY=0

    for _ in $(seq 1 150); do
        if ! kill -0 "$LIVOX_PID" 2>/dev/null; then
            echo "[ERROR] Livox driver exited during startup."
            wait "$LIVOX_PID" || true
            exit 1
        fi

        CURRENT_TYPE="$(
            timeout 2s ros2 topic type "$LIDAR_TOPIC" \
                2>/dev/null || true
        )"

        if [ "$CURRENT_TYPE" = "$EXPECTED_TYPE" ]; then
            READY=1
            break
        fi

        sleep 0.1
    done

    if [ "$READY" -ne 1 ]; then
        echo "[ERROR] Timed out waiting for:"
        echo "        $LIDAR_TOPIC"
        exit 1
    fi
fi

echo "[OK] LiDAR topic type: $EXPECTED_TYPE"

# È·ÈÏ²»Ö»ÊÇ ROS graph ÖÐ´æÔÚ»°Ìâ£¬¶øÊÇÕæµÄÊÕµ½ÁËÒ»ÌõÏûÏ¢¡£
if ! timeout 8s ros2 topic echo \
    "$LIDAR_TOPIC" \
    --once \
    --field header \
    >/dev/null 2>&1; then

    echo "[ERROR] LiDAR topic exists, but no CustomMsg was received."
    exit 1
fi

echo "[OK] Mid-360 data is being published."


# ------------------------------------------------------------
# Æô¶¯ red-rover Mid-360 ºó¶Ë
# ------------------------------------------------------------

cd "$RED_ROVER"

echo "[INFO] Starting Mid-360 backend..."
echo "[INFO] Waiting for start/stop commands on $LIDAR_SOCKET"

"$RED_ROVER/.venv/bin/python" \
    "$RED_ROVER/cli.py" run \
    --config "$CONFIG" \
    --sensor lidar
