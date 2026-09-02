#!/usr/bin/env bash
# Перенаправление на 37L ~100M SFT (бывший 32L-sft пресет).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/jgpt-train-37L-sft.sh" "$@"
