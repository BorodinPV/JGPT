#!/usr/bin/env bash
# =============================================================
# jgpt-train-32L.sh — обучение ~83M (32 слоя) на RTX 3080 10 GB
#
# Отдельный launcher: jgpt-smart.sh при смене пресета сбрасывает
# JGPT_PRESET_NUM_LAYERS обратно на 12 из env/*.env цепочки.
#
# Использование:
#   ./scripts/jgpt-train-32L.sh
#   ./scripts/jgpt-train-32L.sh --data-dir data/books/libru_txt_clean
#   ./scripts/jgpt-train-32L.sh --fresh          # убрать старые чекпоинты
#   JGPT_EPOCHS=40 ./scripts/jgpt-train-32L.sh
#
# Остановка: Ctrl+C (checkpoint сохранится через shutdown hook)
# Resume:    ./scripts/jgpt-train-32L.sh
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_allbooks.log"
ENV_FILE="$ROOT/env/32L-83M.env"
CKPT_DIR="$ROOT/checkpoints/all_books"
CKPT_BACKUP="$ROOT/checkpoints/all_books_prev_backup"

DATA_DIR="${JGPT_DATA_DIR:-data/books/libru_txt_clean}"
DO_FRESH=0
SKIP_BUILD=0

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Обучение JGPT ~83M: 32 слоя, d_model=384, seq=1024, batch=4, accum=4, sampled CE 384.
Пресет VRAM: ${ENV_FILE}

Options:
  --data-dir PATH   каталог с .txt (по умолчанию: data/books/libru_txt_clean)
  --fresh           архивировать checkpoints/all_books/* в all_books_prev_backup
  --no-build        не пересобирать CUDA (build/libjgpt_cuda.so)
  -h, --help        эта справка

Env (поверх пресета):
  JGPT_DATA_DIR           то же, что --data-dir
  JGPT_EPOCHS             число эпох (по умолчанию 20)
  JGPT_FINETUNE=1         новый цикл эпох, веса из чекпоинта
  JGPT_PRESET_NUM_LAYERS  только если меняете геометрию в env/32L-83M.env

Примеры:
  ./scripts/jgpt-train-32L.sh --fresh
  ./scripts/jgpt-train-32L.sh --fresh --data-dir data/books/libru_txt_clean
  JGPT_EPOCHS=80 ./scripts/jgpt-train-32L.sh
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
    echo "[32L] ERROR: missing preset: $ENV_FILE" >&2
    exit 1
fi

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$ROOT/$DATA_DIR"
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

txt_count="$(find "$DATA_DIR" -name '*.txt' -type f 2>/dev/null | wc -l | tr -d ' ')"
if [[ "$txt_count" -eq 0 ]]; then
    echo "[32L] ERROR: no .txt files in $DATA_DIR" >&2
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
    echo "[32L] --fresh: moved $moved file(s) to $CKPT_BACKUP"
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
    echo "[32L] NOTE: found checkpoint in $CKPT_DIR — resume (must be 32 layers / seq ${JGPT_MAX_SEQ_LEN})"
elif [[ "$DO_FRESH" -eq 0 ]]; then
    echo "[32L] NOTE: training from scratch (no checkpoint). Use --fresh if old weights remain."
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo " JGPT Train 32L (~83M)  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo " layers=${JGPT_PRESET_NUM_LAYERS}  seq=${JGPT_MAX_SEQ_LEN}  batch=${JGPT_BATCH_SIZE}"
echo " accum=${JGPT_ACCUMULATION_STEPS}  CE=${JGPT_TRAIN_LOSS_MODE}/${JGPT_SAMPLED_CE_CANDIDATES}"
echo " data=${DATA_DIR}  (${txt_count} txt)"
echo " ckpt=${CKPT_DIR}"
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
        echo "[32L] ERROR: --no-build but no lib in build/" >&2
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
