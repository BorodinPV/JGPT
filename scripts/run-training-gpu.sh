#!/usr/bin/env bash
# Окружение для обучения (MultiBookTrain / TrainLLM → LLMTrainer).
# Переменные зондов (JGPT_BATCH_PROBE*, JGPT_PROBE_*) здесь не задаются — в обучении не используются.
#
# В логе LLMTrainer.train(): «Сводка JGPT_* (обучение)».
#
# Рекомендуемые пресеты:
#
#   # Полный GPU e2e с FP16 и замером производительности:
#   ./scripts/train-e2e-gpu.sh
#
#   # Sampled CE (train-only; быстрее, eval остаётся full-vocab):
#   JGPT_TRAIN_LOSS_MODE=sampled JGPT_CE_ASYNC=0 JGPT_SAMPLED_CE_CANDIDATES=64 \
#     ./scripts/train-e2e-gpu.sh
#
#   # Без пересборки .so (e2e):
#   ./scripts/run-training-gpu.sh e2e
#   ./run-training.sh e2e
#
#   # Отдельные точки входа:
#   ./run-training.sh single|train [аргументы exec.args]
#   ./run-training.sh profile
#
# Переопределение любого флага: VAR=value ./run-training.sh ...
# Для unit-тестов с пустым env используйте mvn без этого скрипта.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
  export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
fi

# exec:java выполняется в JVM Maven — heap задаётся через MAVEN_OPTS (или внешний JAVA_TOOL_OPTIONS).
if [[ -n "${JGPT_JAVA_MEM:-}" ]]; then
  export MAVEN_OPTS="${JGPT_JAVA_MEM} ${MAVEN_OPTS:-}"
fi

# --- JGPT_* (только то, что читает цепочка обучения) ---

export JGPT_ACTIVATION_CACHE_FP16="${JGPT_ACTIVATION_CACHE_FP16:-1}"

export JGPT_BATCH_DIRECT="${JGPT_BATCH_DIRECT:-1}"
export JGPT_BATCH_PINNED="${JGPT_BATCH_PINNED:-1}"
export JGPT_BATCH_PREFETCH="${JGPT_BATCH_PREFETCH:-1}"
export JGPT_BATCH_SIZE="${JGPT_BATCH_SIZE:-}"

# BlockActivationCacheDevice: grow-only, опциональный лимит байт, thread-local пул при смене формы.
export JGPT_BLOCK_CACHE_GROW_ONLY="${JGPT_BLOCK_CACHE_GROW_ONLY:-1}"
# 0 = без лимита; иначе оценка totalFloats × (2 при JGPT_ACTIVATION_CACHE_FP16 иначе 4).
export JGPT_BLOCK_CACHE_MAX_BYTES="${JGPT_BLOCK_CACHE_MAX_BYTES:-0}"
export JGPT_BLOCK_CACHE_POOL="${JGPT_BLOCK_CACHE_POOL:-1}"
export JGPT_BLOCK_CACHE_POOL_MAX="${JGPT_BLOCK_CACHE_POOL_MAX:-2}"

export JGPT_CE_GPU_MIN_ELEMENTS="${JGPT_CE_GPU_MIN_ELEMENTS:-0}"

# Async CE: читает loss после backward+synchronizeStream; совместимо с FP16 matmul.
# ВАЖНО: несовместимо с JGPT_TRAIN_LOSS_MODE=sampled — при sampled выставляйте JGPT_CE_ASYNC=0.
export JGPT_CE_ASYNC="${JGPT_CE_ASYNC:-1}"

# JGPT_TRAIN_LOSS_MODE:
#   full    — полный vocab CE (дефолт; тяжелее, корректнее для мониторинга train loss)
#   sampled — candidate CE только в train loop (быстрее; eval всегда full-vocab)
#             при sampled: JGPT_CE_ASYNC=0, JGPT_SAMPLED_CE_CANDIDATES=64 (рекомендуется)
export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"

export JGPT_CHECKPOINT_ASYNC="${JGPT_CHECKPOINT_ASYNC:-0}"

# Явный путь к .so; если пусто и есть сборка в build/ — подставляется автоматически.
export JGPT_CUDA_LIB="${JGPT_CUDA_LIB:-}"
if [[ -z "$JGPT_CUDA_LIB" && -f "$ROOT/build/libjgpt_cuda.so" ]]; then
  export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
fi

export JGPT_DECODER_GPU_PIPELINE="${JGPT_DECODER_GPU_PIPELINE:-1}"
export JGPT_DEVICE_DECODER_BWD="${JGPT_DEVICE_DECODER_BWD:-1}"
export JGPT_DEVICE_LOGITS_TRAIN="${JGPT_DEVICE_LOGITS_TRAIN:-1}"

# Выйти после N-го шага оптимизатора (0 = без лимита).
# Полезно для профилирования: JGPT_EXIT_AFTER_STEP=20 ./run-training.sh e2e
export JGPT_EXIT_AFTER_STEP="${JGPT_EXIT_AFTER_STEP:-0}"

export JGPT_FP16_MATMUL="${JGPT_FP16_MATMUL:-1}"

# При FP16 matmul loss scale только динамический (JGPT_FP16_DYNAMIC_*).
# Начальный scale: train-e2e-gpu.sh ставит 8192, здесь дефолт 65536 (агрессивнее).
export JGPT_FP16_DYNAMIC_GROWTH_INTERVAL="${JGPT_FP16_DYNAMIC_GROWTH_INTERVAL:-2000}"
export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-65536}"
export JGPT_FP16_DYNAMIC_MAX="${JGPT_FP16_DYNAMIC_MAX:-65536}"

# Один JNI RMSNorm + LM-head matmul при gpuResident; при ошибке Java — откат на раздельный путь.
export JGPT_FUSED_LM_HEAD="${JGPT_FUSED_LM_HEAD:-1}"

export JGPT_FULL_GPU_TRAIN="${JGPT_FULL_GPU_TRAIN:-0}"
export JGPT_GENERATE_GPU_KV="${JGPT_GENERATE_GPU_KV:-1}"
export JGPT_GPU_E2E_TRAIN="${JGPT_GPU_E2E_TRAIN:-1}"

# Пусто — авто (цвет при интерактивной консоли); иначе 0/1.
export JGPT_LOG_COLOR="${JGPT_LOG_COLOR:-}"

# Пусто — без лимита окон на книгу (см. MultiBookTrain).
export JGPT_MAX_SEQUENCES="${JGPT_MAX_SEQUENCES:-}"

export JGPT_PROFILE="${JGPT_PROFILE:-0}"
export JGPT_PROFILE_STEPS="${JGPT_PROFILE_STEPS:-20}"

export JGPT_TIMINGS="${JGPT_TIMINGS:-0}"

# 1 = включить сразу JGPT_PROFILE и JGPT_TIMINGS + подробный PERF на первые 20 шагов.
# Выводит: прямой / лосс+∂CE / обратный / клип+опт / сумма мс / ток/с на каждый шаг.
export JGPT_TRAIN_PERF="${JGPT_TRAIN_PERF:-0}"

export JGPT_TRAIN_GPU_RESIDENT="${JGPT_TRAIN_GPU_RESIDENT:-1}"

# --- Обработка необязательных префиксов ---
# e2e: включает полный GPU-шаг (JGPT_FULL_GPU_TRAIN). Можно повторять: e2e e2e → ок.
while [[ "${1:-}" == e2e ]]; do
  shift
  export JGPT_GPU_E2E_TRAIN=1
  export JGPT_FULL_GPU_TRAIN=1
done

case "${1:-}" in
  single|train)
    shift
    if [[ $# -gt 0 ]]; then
      set -- mvn -q compile exec:java \
        -Dexec.mainClass=com.veles.llm.jgpt.app.TrainLLM \
        -Dexec.args="$*"
    else
      set -- mvn -q compile exec:java \
        -Dexec.mainClass=com.veles.llm.jgpt.app.TrainLLM
    fi
    ;;
  profile)
    shift
    if [[ $# -gt 0 ]]; then
      set -- mvn -q compile exec:java \
        -Dexec.mainClass=com.veles.llm.jgpt.app.ProfileQuickRun \
        -Dexec.args="$*"
    else
      set -- mvn -q compile exec:java \
        -Dexec.mainClass=com.veles.llm.jgpt.app.ProfileQuickRun
    fi
    ;;
esac

if [[ $# -eq 0 ]]; then
  set -- mvn -q compile exec:java \
    -Dexec.mainClass=com.veles.llm.jgpt.app.MultiBookTrain \
    -Dexec.args="--boo ."
fi

exec "$@"
