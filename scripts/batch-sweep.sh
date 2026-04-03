#!/usr/bin/env bash
# Sweep batch sizes and report throughput from first timing/profiler step.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BATCH_SIZES="${BATCH_SIZES:-4 8 9 12 16 24 32}"
STEPS="${SWEEP_STEPS:-50}"
CMD="${SWEEP_CMD:-./run-training.sh profile}"
OUT_DIR="${SWEEP_OUT_DIR:-$ROOT/batch-sweep-logs}"

mkdir -p "$OUT_DIR"

echo "Batch sweep command: $CMD"
echo "Batch sizes: $BATCH_SIZES"
echo "Stop step: $STEPS"
echo
printf "%-6s %-9s %-8s %s\n" "batch" "tok/s" "sigma" "log"

best_batch=""
best_tokps=0
best_sigma=""

for batch in $BATCH_SIZES; do
  log="$OUT_DIR/batch-${batch}.log"
  if ! JGPT_BATCH_SIZE="$batch" JGPT_TIMINGS=1 JGPT_PROFILE=1 JGPT_PROFILE_STEPS="$STEPS" JGPT_EXIT_AFTER_STEP="$STEPS" bash -lc "$CMD" >"$log" 2>&1; then
    printf "%-6s %-9s %-8s %s\n" "$batch" "FAILED" "-" "$log"
    continue
  fi

  line="$(awk '
    /ток\/с≈|токенов\/с ≈/ { x = $0 }
    END { print x }
  ' "$log")"
  if [[ -z "$line" ]]; then
    printf "%-6s %-9s %-8s %s\n" "$batch" "NO_METRIC" "-" "$log"
    continue
  fi

  sigma="$(awk '
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
    }
  ' "$log")"
  tokps="$(awk '
    /токенов\/с ≈/ {
      for (i = NF; i >= 1; i--) {
        if ($i ~ /^[0-9]+\.?[0-9]*$/) {
          print $i
          exit
        }
      }
    }
    /ток\/с≈/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^ток\/с≈[0-9]/) {
          v = $i
          sub(/^ток\/с≈/, "", v)
          gsub(",", ".", v)
          print v
          exit
        }
      }
    }
  ' "$log")"

  if [[ -z "$sigma" || -z "$tokps" ]]; then
    printf "%-6s %-9s %-8s %s\n" "$batch" "PARSE_ERR" "-" "$log"
    continue
  fi

  printf "%-6s %-9s %-8s %s\n" "$batch" "$tokps" "${sigma}ms" "$log"

  if awk -v a="$tokps" -v b="$best_tokps" 'BEGIN { exit !(a > b) }'; then
    best_tokps="$tokps"
    best_batch="$batch"
    best_sigma="$sigma"
  fi
done

echo
if [[ -n "$best_batch" ]]; then
  echo "Best: batch=${best_batch}, tok/s=${best_tokps}, Σ=${best_sigma}ms"
else
  echo "No successful runs."
  exit 2
fi
