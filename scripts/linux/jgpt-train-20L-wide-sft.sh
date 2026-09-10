#!/usr/bin/env bash
# =============================================================
# jgpt-train-20L-wide-sft.sh — SFT after 20L-wide pretrain
# Checkpoints: checkpoints/wide_20L_sft/  Tokenizer: tokenizer_wide_16k.bin
#
# Usage:
#   ./scripts/linux/jgpt-train-20L-wide-sft.sh --no-build
# Windows: .\scripts\windows\jgpt-train-20L-wide-sft.cmd --no-build
# Chat:    ./scripts/linux/jgpt-chat.sh --boo . --layers 20 --seq-len 1024 \
#            --model checkpoints/wide_20L_sft/model_best.bin \
#            --tokenizer checkpoints/tokenizer_wide_16k.bin
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_20L_wide_sft.log"
ENV_FILE="$ROOT/env/20L-wide-sft.env"
CKPT_DIR="$ROOT/checkpoints/wide_20L_sft"
CKPT_BACKUP="$ROOT/checkpoints/wide_20L_sft_prev_backup"
TOKENIZER_FILE="$ROOT/checkpoints/tokenizer_wide_16k.bin"
SRC_BEST="$ROOT/checkpoints/wide_20L_16k_1024/model_best.bin"
SRC_FINAL="$ROOT/checkpoints/wide_20L_16k_1024/model_final.bin"

DATA_DIR="${JGPT_DATA_DIR:-data/sft/exam}"
DO_FRESH=0
SKIP_BUILD=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

SFT after 20L-wide pretrain (one dialog per window). Data: data/sft/exam
Preset: ${ENV_FILE}
Checkpoints: checkpoints/wide_20L_sft

Options:
  --data-dir PATH   каталог с .jsonl (по умолчанию: data/sft/exam)
  --fresh           архивировать только wide_20L_sft (tokenizer не трогать)
  --no-build        не пересобирать CUDA
  -h, --help        эта справка

Примеры:
  ./scripts/linux/jgpt-train-20L-wide-sft.sh --no-build
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
    echo "[20L-WIDE-SFT] ERROR: missing preset: $ENV_FILE" >&2
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
    echo "[20L-WIDE-SFT] generating exam JSONL -> $DATA_DIR"
    python3 "$ROOT/scripts/sft-make-exam.py" --dst "$DATA_DIR/exam.jsonl"
    jsonl_count="$(find "$DATA_DIR" -name '*.jsonl' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${jsonl_count:-0}" -eq 0 ]]; then
    echo "[20L-WIDE-SFT] ERROR: no .jsonl files in $DATA_DIR (JGPT_SFT=1)" >&2
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
    echo "[20L-WIDE-SFT] --fresh: moved $moved file(s) to $CKPT_BACKUP (tokenizer untouched)"
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
    echo "[20L-WIDE-SFT] NOTE: found Adam checkpoint in $CKPT_DIR — resume"
else
    mkdir -p "$CKPT_DIR"
    if [[ ! -f "$CKPT_DIR/model_final.bin" ]]; then
        if [[ ! -f "$SRC_BEST" && ! -f "$SRC_FINAL" ]]; then
            echo "[20L-WIDE-SFT] ERROR: missing seed weights: $SRC_BEST" >&2
            echo "  Run pretrain first: ./scripts/linux/jgpt-train-20L-wide.sh --no-build" >&2
            exit 1
        fi
        seed="$SRC_BEST"
        [[ -f "$seed" ]] || seed="$SRC_FINAL"
        cp "$seed" "$CKPT_DIR/model_final.bin"
        echo "[20L-WIDE-SFT] seeded weights: $seed -> $CKPT_DIR/model_final.bin (fresh Adam)"
    else
        echo "[20L-WIDE-SFT] NOTE: model_final.bin exists, no Adam checkpoint — continue from weights, step 0"
    fi
fi
if [[ ! -f "$TOKENIZER_FILE" ]]; then
    echo "[20L-WIDE-SFT] ERROR: missing tokenizer: $TOKENIZER_FILE" >&2
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo " JGPT Train 20L-wide SFT  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo " layers=${JGPT_PRESET_NUM_LAYERS}  d=${JGPT_D_MODEL}  heads=${JGPT_NUM_HEADS}  seq=${JGPT_MAX_SEQ_LEN}  vocab=${JGPT_VOCAB_SIZE}"
echo " batch=${JGPT_BATCH_SIZE}  accum=${JGPT_ACCUMULATION_STEPS}  eff_batch=$((JGPT_BATCH_SIZE * JGPT_ACCUMULATION_STEPS))  lr=${JGPT_LEARNING_RATE:-}  pack=${JGPT_SFT_PACK:-}"
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
        echo "[20L-WIDE-SFT] ERROR: --no-build but no lib in build/" >&2
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
