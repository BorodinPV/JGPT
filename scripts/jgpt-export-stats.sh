#!/usr/bin/env bash
# =============================================================
# jgpt-export-stats.sh — Генерирует state/stats.json для дашборда
#
# Использование:
#   ./scripts/jgpt-export-stats.sh
#   watch -n 30 ./scripts/jgpt-export-stats.sh
# =============================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/jgpt-export-stats.py" "$ROOT"
