#!/usr/bin/env bash
# Thin operator wrapper: local src first, then CLI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m geodude_teleoperation.teleop.ur_stringer_teleop "$@"
