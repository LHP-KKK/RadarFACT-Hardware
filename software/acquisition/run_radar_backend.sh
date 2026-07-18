#!/usr/bin/env bash
set -euo pipefail

RED_ROVER="${RED_ROVER_ROOT:-$HOME/red-rover/collect}"
CONFIG="${RADAR_CONFIG:-config/custom/grt-i-demo-full.yaml}"

cd "$RED_ROVER"

exec "$RED_ROVER/.venv/bin/python" \
  "$RED_ROVER/cli.py" run \
  --config "$CONFIG" \
  --sensor radar
