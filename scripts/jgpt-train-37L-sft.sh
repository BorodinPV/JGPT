#!/usr/bin/env bash
# =============================================================
# jgpt-train-37L-sft.sh — ~100.4M: 37 слоёв, vocab 16k, seq 2048, SFT
#
# С нуля: BPE checkpoints/tokenizer_sft_16k.bin и
# checkpoints/sft_37L_16k_2048/. Не jgpt-smart.sh.
#
# Использование:
#   ./scripts/jgpt-train-37L-sft.sh
#   ./scripts/jgpt-train-37L-sft.sh --fresh
#   JGPT_BATCH_SIZE=2 JGPT_ACCUMULATION_STEPS=64 ./scripts/jgpt-train-37L-sft.sh --no-build
#
# Остановка: Ctrl+C
# Resume:    ./scripts/jgpt-train-37L-sft.sh
# Чат:       set -a; source env/37L-sft-100M.env; set +a
#            ./scripts/jgpt-chat.sh --boo . --layers 37 --seq-len 2048 \
#              --model checkpoints/sft_37L_16k_2048/model_best.bin \
#              --tokenizer checkpoints/tokenizer_sft_16k.bin
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_sft_37L.log"
ENV_FILE="$ROOT/env/37L-sft-100M.env"
CKPT_DIR="$ROOT/checkpoints/sft_37L_16k_2048"
CKPT_BACKUP="$ROOT/checkpoints/sft_37L_16k_2048_prev_backup"
TOKENIZER_FILE="$ROOT/checkpoints/tokenizer_sft_16k.bin"

DATA_DIR="${JGPT_DATA_DIR:-data/sft/raw}"
DO_FRESH=0
SKIP_BUILD=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Обучение JGPT ~100.4M SFT: 37 слоёв, seq=2048, vocab=16000, JSONL
(лосс только на ответах ассистента). Данные: data/sft/raw
Пресет: ${ENV_FILE}
batch=4, accum=32 (эффективный батч 128), LR=1e-3, sampled CE 384.

Options:
  --data-dir PATH   каталог с .jsonl (по умолчанию: data/sft/raw)
  --fresh           архивировать чекпоинты пресета и tokenizer_sft_16k.bin
  --no-build        не пересобирать CUDA
  -h, --help        эта справка

Env (поверх пресета):
  JGPT_DATA_DIR           то же, что --data-dir
  JGPT_EPOCHS             число эпох (по умолчанию 20)
  JGPT_BATCH_SIZE         если OOM — 2 (тогда accum=64, те же 128 окон)
  JGPT_ACCUMULATION_STEPS по умолчанию 32
  JGPT_LEARNING_RATE      по умолчанию 0.001
  JGPT_MAX_SEQ_LEN        если OOM — 1536

Примеры:
  ./scripts/jgpt-train-37L-sft.sh
  JGPT_BATCH_SIZE=2 JGPT_ACCUMULATION_STEPS=64 ./scripts/jgpt-train-37L-sft.sh --no-build
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --fresh)
            DO_FRESH=1
            shift
            ;;
        --no-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[37L-SFT] ERROR: missing preset: $ENV_FILE" >&2
    exit 1
fi

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$ROOT/$DATA_DIR"
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

if [[ -f "$ROOT/scripts/sft-export-jsonl.py" ]]; then
    echo "[37L-SFT] parquet → jsonl (если ещё нет)..."
    PYTHONPATH="/tmp/jgpt-pyarrow${PYTHONPATH:+:$PYTHONPATH}" \
        python3 "$ROOT/scripts/sft-export-jsonl.py" || echo "[37L-SFT] WARN: parquet export skipped" >&2
fi

jsonl_count="$(find "$DATA_DIR" -name '*.jsonl' -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$jsonl_count" -eq 0 ]]; then
    echo "[37L-SFT] ERROR: no .jsonl files in $DATA_DIR (JGPT_SFT=1)" >&2
    exit 1
fi

set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
export JGPT_DATA_DIR="$DATA_DIR"
set +a

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-25-openjdk-amd64}"
if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
    export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
fi
if [[ -n "${JGPT_JAVA_MEM:-}" ]]; then
    export MAVEN_OPTS="${JGPT_JAVA_MEM} ${MAVEN_OPTS:-}"
fi

export JGPT_IF_STEP_BEYOND_PLAN="${JGPT_IF_STEP_BEYOND_PLAN:-restart_schedule}"

if [[ "$DO_FRESH" -eq 1 ]]; then
    mkdir -p "$CKPT_BACKUP"
    shopt -s nullglob
    moved=0
    for f in "$CKPT_DIR"/*; do
        mv "$f" "$CKPT_BACKUP/"
        moved=$((moved + 1))
    done
    shopt -u nullglob
    if [[ -f "$TOKENIZER_FILE" ]]; then
        mv "$TOKENIZER_FILE" "$CKPT_BACKUP/"
        moved=$((moved + 1))
    fi
    echo "[37L-SFT] --fresh: moved $moved file(s) to $CKPT_BACKUP"
fi

has_ckpt=0
[[ -f "$CKPT_DIR/checkpoint_final.bin" ]] && has_ckpt=1
shopt -s nullglob
for _ck in "$CKPT_DIR"/checkpoint_epoch_*.bin; do
    has_ckpt=1
    break
done
shopt -u nullglob
if [[ "$has_ckpt" -eq 1 ]]; then
    echo "[37L-SFT] NOTE: found checkpoint in $CKPT_DIR — resume (37L / seq ${JGPT_MAX_SEQ_LEN} / vocab ${JGPT_VOCAB_SIZE})"
elif [[ -f "$TOKENIZER_FILE" ]]; then
    echo "[37L-SFT] NOTE: tokenizer exists, weights from scratch: $TOKENIZER_FILE"
else
    echo "[37L-SFT] NOTE: training from scratch, will train BPE vocab=${JGPT_VOCAB_SIZE}"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo " JGPT Train 37L SFT (~100.4M)  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo " layers=${JGPT_PRESET_NUM_LAYERS}  seq=${JGPT_MAX_SEQ_LEN}  vocab=${JGPT_VOCAB_SIZE}  batch=${JGPT_BATCH_SIZE}"
echo " accum=${JGPT_ACCUMULATION_STEPS}  eff_batch=$((JGPT_BATCH_SIZE * JGPT_ACCUMULATION_STEPS))  lr=${JGPT_LEARNING_RATE:-}  CE=${JGPT_TRAIN_LOSS_MODE}/${JGPT_SAMPLED_CE_CANDIDATES}"
echo " data=${DATA_DIR}  (${jsonl_count} jsonl)"
echo " ckpt=${CKPT_DIR}"
echo " tok=${TOKENIZER_FILE}"
echo " log=${LOG_FILE}"
echo "════════════════════════════════════════════════════════════"
echo ""

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    bash "$ROOT/scripts/build-cuda.sh"
else
    if [[ -f "$ROOT/build/libjgpt_cuda.so" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    elif [[ -f "$ROOT/build/jgpt_cuda.dll" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/jgpt_cuda.dll"
    else
        echo "[37L-SFT] ERROR: --no-build but no lib in build/" >&2
        exit 1
    fi
fi

mvn -q compile
CP_FILE="$(mktemp)"
trap 'rm -f "$CP_FILE"' EXIT
mvn -q dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile="$CP_FILE"
CP="$ROOT/target/classes:$(tr -d '\r\n' < "$CP_FILE")"

java ${MAVEN_OPTS:-} \
    --sun-misc-unsafe-memory-access=allow \
    --add-modules=jdk.incubator.vector \
    --enable-preview \
    -cp "$CP" \
    com.veles.llm.jgpt.app.AllBooksTrain \
    --boo "$ROOT" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee -a "$LOG_FILE"
