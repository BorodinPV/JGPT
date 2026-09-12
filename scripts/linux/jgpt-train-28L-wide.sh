#!/usr/bin/env bash
# =============================================================
# jgpt-train-28L-wide.sh — books LM pretrain, 28L / d=512 / seq 1024
# Checkpoints: checkpoints/wide_28L_16k_1024/  Tokenizer: tokenizer_wide_16k.bin
# Does NOT touch sft_37L_*.
#
# Usage:
#   ./scripts/linux/jgpt-train-28L-wide.sh --no-build
# Windows: .\scripts\windows\jgpt-train-28L-wide.cmd --no-build
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_28L_wide.log"
ENV_FILE="$ROOT/env/28L-wide-pretrain.env"
CKPT_DIR="$ROOT/checkpoints/wide_28L_16k_1024"
CKPT_BACKUP="$ROOT/checkpoints/wide_28L_16k_1024_prev_backup"
TOKENIZER_FILE="$ROOT/checkpoints/tokenizer_wide_16k.bin"

DATA_DIR="${JGPT_DATA_DIR:-data/books/pretrain_txt}"
DO_FRESH=0
SKIP_BUILD=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Books LM pretrain (28L, d_model=512, seq=1024, vocab=16000).
Preset: ${ENV_FILE}
Checkpoints: checkpoints/wide_28L_16k_1024

Options:
  --data-dir PATH   каталог с .txt (по умолчанию: data/books/pretrain_txt)
  --fresh           архивировать только wide_28L_16k_1024 (tokenizer не трогать)
  --no-build        не пересобирать CUDA
  -h, --help        эта справка

Stop: touch state/STOP  (then wait for checkpoint_final; Ctrl+C is OK on Linux)

Примеры:
  ./scripts/linux/jgpt-train-28L-wide.sh --no-build
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
    echo "[28L-WIDE] ERROR: missing preset: $ENV_FILE" >&2
    exit 1
fi

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$ROOT/$DATA_DIR"
fi

txt_count=0
if [[ -d "$DATA_DIR" ]]; then
    txt_count="$(find "$DATA_DIR" -name '*.txt' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${txt_count:-0}" -eq 0 ]]; then
    echo "[28L-WIDE] no .txt in $DATA_DIR — fetching ru.wikipedia starter corpus"
    python3 "$ROOT/scripts/fetch-ru-pretrain.py" --dst "$DATA_DIR"
    txt_count="$(find "$DATA_DIR" -name '*.txt' -type f 2>/dev/null | wc -l | tr -d ' ')"
fi
if [[ "${txt_count:-0}" -eq 0 ]]; then
    echo "[28L-WIDE] ERROR: no .txt files in $DATA_DIR" >&2
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
    echo "[28L-WIDE] --fresh: moved $moved file(s) to $CKPT_BACKUP (tokenizer untouched)"
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
    echo "[28L-WIDE] NOTE: found Adam checkpoint in $CKPT_DIR — resume"
else
    mkdir -p "$CKPT_DIR"
    if [[ -f "$CKPT_DIR/model_final.bin" ]]; then
        echo "[28L-WIDE] NOTE: model_final.bin exists, no Adam checkpoint — continue from weights, step 0"
    else
        echo "[28L-WIDE] from-scratch pretrain (no seed weights)"
    fi
fi
if [[ -f "$TOKENIZER_FILE" ]]; then
    echo "[28L-WIDE] tokenizer exists: $TOKENIZER_FILE"
else
    echo "[28L-WIDE] tokenizer missing — AllBooksTrain will train BPE on the corpus"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo " JGPT Train 28L-wide pretrain  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo " layers=${JGPT_PRESET_NUM_LAYERS}  d=${JGPT_D_MODEL}  heads=${JGPT_NUM_HEADS}  seq=${JGPT_MAX_SEQ_LEN}  vocab=${JGPT_VOCAB_SIZE}"
echo " batch=${JGPT_BATCH_SIZE}  accum=${JGPT_ACCUMULATION_STEPS}  eff_batch=$((JGPT_BATCH_SIZE * JGPT_ACCUMULATION_STEPS))  lr=${JGPT_LEARNING_RATE:-}"
echo " data=${DATA_DIR}  (${txt_count} txt)"
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
        echo "[28L-WIDE] ERROR: --no-build but no lib in build/" >&2
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
