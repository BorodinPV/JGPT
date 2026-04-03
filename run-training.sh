#!/usr/bin/env bash
# Совместимость: раньше запускали ./run-training.sh — делегирует scripts/run-training-gpu.sh
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/scripts/run-training-gpu.sh" "$@"
