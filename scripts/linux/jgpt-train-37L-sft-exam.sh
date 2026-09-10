#!/usr/bin/env bash
# =============================================================
# jgpt-train-37L-sft-exam.sh — tiny clean exam finetune from short-ft model_best.bin
# Checkpoints: checkpoints/sft_37L_exam/ (not 20-epoch 37L, does not overwrite short_ft).
#
# Usage:
#   ./scripts/linux/jgpt-train-37L-sft-exam.sh --no-build
# Windows: .\scripts\windows\jgpt-train-37L-sft-exam.cmd --no-build
#
# Остановка: Ctrl+C
# Resume:    ./scripts/linux/jgpt-train-37L-sft-exam.sh --no-build
# Чат:       ./scripts/linux/jgpt-chat.sh --boo . --layers 37 --seq-len 2048 \
#              --model checkpoints/sft_37L_exam/model_best.bin \
#              --tokenizer checkpoints/tokenizer_sft_16k.bin
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_sft_37L_exam.log"
ENV_FILE="$ROOT/env/37L-sft-exam.env"
CKPT_DIR="$ROOT/checkpoints/sft_37L_exam"
CKPT_BACKUP="$ROOT/checkpoints/sft_37L_exam_prev_backup"
TOKENIZER_FILE="$ROOT/checkpoints/tokenizer_sft_16k.bin"
SRC_BEST="$ROOT/checkpoints/sft_37L_short_ft/model_best.bin"

DATA_DIR="${JGPT_DATA_DIR:-data/sft/exam}"
DO_FRESH=0
SKIP_BUILD=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Tiny clean exam finetune (capitals, 2+2, yes/no). Data: data/sft/exam
Preset: ${ENV_FILE}
Checkpoints: checkpoints/sft_37L_exam

Options:
  --data-dir PATH   каталог с .jsonl (по умолчанию: data/sft/exam)
  --fresh           архивировать только sft_37L_exam (tokenizer не трогать)
  --no-build        не пересобирать CUDA
  -h, --help        эта справка

Примеры:
  ./scripts/linux/jgpt-train-37L-sft-exam.sh --no-build
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
    echo "[37L-SFT-EXAM] ERROR: missing preset: $ENV_FILE" >&2
    exit 1
fi

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$ROOT/$DATA_DIR"
fi

jsonl_count=0
if [[ -d "$DATA_DIR" ]]; then
    jsonl_count="$(find "$DATA_DIR" -name '*.jsonl' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${jsonl_count:-0}" -eq 0 ]]; then
    echo "[37L-SFT-EXAM] generating exam JSONL -> $DATA_DIR"
    python3 "$ROOT/scripts/sft-make-exam.py" --dst "$DATA_DIR/exam.jsonl"
    jsonl_count="$(find "$DATA_DIR" -name '*.jsonl' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${jsonl_count:-0}" -eq 0 ]]; then
    echo "[37L-SFT-EXAM] ERROR: no .jsonl files in $DATA_DIR (JGPT_SFT=1)" >&2
    exit 1
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

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
    echo "[37L-SFT-EXAM] --fresh: moved $moved file(s) to $CKPT_BACKUP (tokenizer untouched)"
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
    echo "[37L-SFT-EXAM] NOTE: found Adam checkpoint in $CKPT_DIR — resume"
else
    mkdir -p "$CKPT_DIR"
    if [[ ! -f "$CKPT_DIR/model_final.bin" ]]; then
        if [[ ! -f "$SRC_BEST" ]]; then
            echo "[37L-SFT-EXAM] ERROR: missing seed weights: $SRC_BEST" >&2
            exit 1
        fi
        cp "$SRC_BEST" "$CKPT_DIR/model_final.bin"
        echo "[37L-SFT-EXAM] seeded weights: $SRC_BEST -> $CKPT_DIR/model_final.bin (fresh Adam)"
    else
        echo "[37L-SFT-EXAM] NOTE: model_final.bin exists, no Adam checkpoint — continue from weights, step 0"
    fi
fi
if [[ ! -f "$TOKENIZER_FILE" ]]; then
    echo "[37L-SFT-EXAM] ERROR: missing tokenizer: $TOKENIZER_FILE" >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo " JGPT Train 37L SFT exam-ft  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo " layers=${JGPT_PRESET_NUM_LAYERS}  seq=${JGPT_MAX_SEQ_LEN}  vocab=${JGPT_VOCAB_SIZE}  batch=${JGPT_BATCH_SIZE}"
echo " accum=${JGPT_ACCUMULATION_STEPS}  eff_batch=$((JGPT_BATCH_SIZE * JGPT_ACCUMULATION_STEPS))  lr=${JGPT_LEARNING_RATE:-}  CE=${JGPT_TRAIN_LOSS_MODE}/${JGPT_SAMPLED_CE_CANDIDATES}"
echo " data=${DATA_DIR}  (${jsonl_count} jsonl)"
echo " ckpt=${CKPT_DIR}"
echo " tok=${TOKENIZER_FILE}"
echo " log=${LOG_FILE}"
echo "════════════════════════════════════════════════════════════"
echo ""

if [[ "$SKIP_BUILD" -eq 0 ]]; then
    bash "$ROOT/scripts/linux/build-cuda.sh"
else
    if [[ -f "$ROOT/build/libjgpt_cuda.so" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    elif [[ -f "$ROOT/build/jgpt_cuda.dll" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/jgpt_cuda.dll"
    else
        echo "[37L-SFT-EXAM] ERROR: --no-build but no lib in build/" >&2
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
