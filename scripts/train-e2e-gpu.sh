#!/usr/bin/env bash
# Сборка libjgpt_cuda.so и запуск полного GPU e2e-обучения (как вручную: cmake + JGPT_CUDA_LIB + run-training.sh e2e).
#
# Каталог: скрипт сам делает cd в корень репозитория. Если приглашение shell уже «…/JGPT$», не выполняйте
# «cd Рабочий стол/JGPT» без кавычек и полного пути — иначе bash ищет вложенную папку и выдаёт «Нет такого файла».
#
# Использование (из корня репозитория):
#   ./scripts/train-e2e-gpu.sh
#   ./scripts/train-e2e-gpu.sh single --boo …
#   JGPT_FP16_DYNAMIC_INITIAL=8192 ./scripts/train-e2e-gpu.sh
#
# Переопределение «встроенных» флагов: задайте переменную до вызова (см. DEFAULT ниже).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmake -B build -S src/main/cpp
cmake --build build

export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"

# DEFAULT для этого сценария (в run-training-gpu.sh часть из них иначе):
export JGPT_TRAIN_PERF="${JGPT_TRAIN_PERF:-1}"
export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"
export JGPT_DECODER_LAYER_CUDA_GRAPH="${JGPT_DECODER_LAYER_CUDA_GRAPH:-1}"
export JGPT_FUSED_FFN_RMS_W1W3="${JGPT_FUSED_FFN_RMS_W1W3:-1}"
export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-8192}"
export JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK="${JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK:-256}"
# По умолчанию без сброса scale на каждой эпохе (стабильнее; см. README «FP16»).
export JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH="${JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH:-0}"

# При желании: сброс scale в начале каждой эпохи (может дать лишние overflow в начале эпохи):
#   JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH=1 ./scripts/train-e2e-gpu.sh
# Отключить recovery после серии overflow на min scale:
#   JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK=0 ./scripts/train-e2e-gpu.sh

exec "$ROOT/run-training.sh" e2e "$@"
