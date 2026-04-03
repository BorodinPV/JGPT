#!/usr/bin/env bash
# Запись таймлайна Nsight Systems (CUDA + OS) на несколько секунд обучения.
# Требуется: nsys из NVIDIA Nsight Systems (пакет nsight-systems / nsight-systems-cli).
#
# Примеры:
#   ./scripts/profile-nsys.sh
#   ./scripts/profile-nsys.sh single
#   NSYS_OUT=/tmp/my ./scripts/profile-nsys.sh
#
# Отчёт: jgpt-profile.nsys-rep в корне проекта (или $NSYS_OUT без суффикса).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v nsys &>/dev/null; then
  echo "nsys не найден. Установите Nsight Systems (например пакет nsight-systems) и добавьте в PATH."
  exit 1
fi

OUT="${NSYS_OUT:-$ROOT/jgpt-profile}"
# nsys добавляет расширение к -o
echo "Запись в: ${OUT}.* (остановите Ctrl+C через 5–15 с или дождитесь конца)"
echo "Просмотр: nsys-ui ${OUT}.nsys-rep   (или импорт в Nsight Systems GUI)"

exec nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  -o "$OUT" \
  "$ROOT/run-training.sh" "$@"
