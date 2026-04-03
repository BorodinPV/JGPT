#!/usr/bin/env bash
# Лёгкий perf-guard: короткий прогон и извлечение первого окна JGPT_TIMINGS.
# По умолчанию запускает ./run-training.sh profile (быстро и безопасно по памяти).
#
# Примеры:
#   ./scripts/perf-smoke.sh
#   PERF_SIGMA_MAX_MS=80 ./scripts/perf-smoke.sh
#   PERF_CMD="./run-training.sh single" ./scripts/perf-smoke.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_LOG="${PERF_OUT_LOG:-$ROOT/perf-smoke.log}"
CMD="${PERF_CMD:-./run-training.sh profile}"
MAX_SIGMA_MS="${PERF_SIGMA_MAX_MS:-}"

echo "Perf smoke command: $CMD"
echo "Log: $OUT_LOG"

# shellcheck disable=SC2086
JGPT_TIMINGS=1 $CMD >"$OUT_LOG" 2>&1

LINE="$(awk '/ток\/с≈|токенов\/с ≈/{print; exit}' "$OUT_LOG")"
if [[ -z "${LINE}" ]]; then
  echo "ERROR: timing line (ток/с) not found. See $OUT_LOG"
  exit 2
fi

echo "$LINE"

SIGMA="$(awk '
  /сумма:/ {
    gsub(/,/, ".", $0)
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^[0-9]+\.?[0-9]*$/ && i < NF && $(i+1) ~ /^мс/) {
        print $i
        exit
      }
    }
  }
  /средн\. Σ фаз≈/ {
    sub(/^.*средн\. Σ фаз≈/, "", $0)
    sub(/ мс.*/, "", $0)
    gsub(/,/, ".", $0)
    if ($0 ~ /^[0-9]+\.?[0-9]*$/) {
      print $0
      exit
    }
  }
  /сумма=[0-9]/ {
    sub(/^.*сумма=/, "", $0)
    sub(/[^0-9.].*/, "", $0)
    gsub(/,/, ".", $0)
    if ($0 ~ /^[0-9]+\.?[0-9]*$/) {
      print $0
      exit
    }
  }' "$OUT_LOG")"

if [[ -z "${SIGMA}" ]]; then
  echo "ERROR: failed to parse Σ value. See $OUT_LOG"
  exit 3
fi

SIGMA="${SIGMA/,/.}"
echo "Parsed Σ=${SIGMA} ms"

if [[ -n "$MAX_SIGMA_MS" ]]; then
  awk -v s="$SIGMA" -v max="$MAX_SIGMA_MS" 'BEGIN { exit !(s > max) }'
  if [[ $? -eq 0 ]]; then
    echo "FAIL: Σ=${SIGMA} ms is above PERF_SIGMA_MAX_MS=${MAX_SIGMA_MS}"
    exit 4
  fi
  echo "PASS: Σ=${SIGMA} ms <= PERF_SIGMA_MAX_MS=${MAX_SIGMA_MS}"
fi

