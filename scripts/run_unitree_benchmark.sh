#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TASKS="${1:-all}"
INSTANCES="${2:-0}"
EPISODES="${3:-1}"

python3 benchmark_main.py \
  --headless \
  --device cpu \
  --tasks "${TASKS}" \
  --instances "${INSTANCES}" \
  --episodes "${EPISODES}" \
  --policy zero
