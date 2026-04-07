#!/usr/bin/env bash
# =============================================================
# jgpt-smart.sh — Адаптивное обучение с авто-переключением пресетов
#
# Запускает обучение, следит за логом и автоматически:
#   - понижает пресет при OOM / зависании FP16
#   - повышает пресет обратно после стабильной работы
#   - делает resume после каждого переключения
#
# Иерархия пресетов (от быстрого к безопасному):
#   00-max-throughput → 01-aggressive → 02-stable → 03-recovery → 04-minimal
#
# Использование:
#   ./scripts/jgpt-smart.sh                    # с текущего пресета
#   ./scripts/jgpt-smart.sh 01-aggressive      # явный стартовый пресет
#   Ctrl+C — остановить (checkpoint сохраняется через shutdown hook)
#
# Сборка CUDA + env + Maven: функции jgpt__* / jgpt_* ниже в этом файле.
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ─── Maven / CUDA / env (раньше jgpt-gpu-train-lib.sh) ───────

jgpt__maven_opts() {
    if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
        export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
    fi
    if [[ -n "${JGPT_JAVA_MEM:-}" ]]; then
        export MAVEN_OPTS="${JGPT_JAVA_MEM} ${MAVEN_OPTS:-}"
    fi
}

jgpt__export_train_env() {
    export JGPT_ACTIVATION_CACHE_FP16="${JGPT_ACTIVATION_CACHE_FP16:-1}"

    export JGPT_BATCH_DIRECT="${JGPT_BATCH_DIRECT:-1}"
    export JGPT_BATCH_PINNED="${JGPT_BATCH_PINNED:-1}"
    export JGPT_BATCH_PREFETCH="${JGPT_BATCH_PREFETCH:-1}"
    export JGPT_BATCH_SIZE="${JGPT_BATCH_SIZE:-}"

    export JGPT_BLOCK_CACHE_GROW_ONLY="${JGPT_BLOCK_CACHE_GROW_ONLY:-1}"
    export JGPT_BLOCK_CACHE_MAX_BYTES="${JGPT_BLOCK_CACHE_MAX_BYTES:-0}"
    export JGPT_BLOCK_CACHE_POOL="${JGPT_BLOCK_CACHE_POOL:-1}"
    export JGPT_BLOCK_CACHE_POOL_MAX="${JGPT_BLOCK_CACHE_POOL_MAX:-2}"

    export JGPT_CE_GPU_MIN_ELEMENTS="${JGPT_CE_GPU_MIN_ELEMENTS:-0}"
    export JGPT_CE_ASYNC="${JGPT_CE_ASYNC:-1}"

    export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
    export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
    export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"

    export JGPT_CHECKPOINT_ASYNC="${JGPT_CHECKPOINT_ASYNC:-0}"

    export JGPT_CUDA_LIB="${JGPT_CUDA_LIB:-}"
    if [[ -z "$JGPT_CUDA_LIB" && -f "$ROOT/build/libjgpt_cuda.so" ]]; then
        export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    fi

    export JGPT_DECODER_GPU_PIPELINE="${JGPT_DECODER_GPU_PIPELINE:-1}"
    export JGPT_DEVICE_DECODER_BWD="${JGPT_DEVICE_DECODER_BWD:-1}"
    export JGPT_DEVICE_LOGITS_TRAIN="${JGPT_DEVICE_LOGITS_TRAIN:-1}"
    export JGPT_DECODER_LAYER_CUDA_GRAPH="${JGPT_DECODER_LAYER_CUDA_GRAPH:-1}"

    export JGPT_EXIT_AFTER_STEP="${JGPT_EXIT_AFTER_STEP:-0}"

    export JGPT_FP16_MATMUL="${JGPT_FP16_MATMUL:-1}"
    export JGPT_FP16_DYNAMIC_GROWTH_INTERVAL="${JGPT_FP16_DYNAMIC_GROWTH_INTERVAL:-2000}"
    export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-65536}"
    export JGPT_FP16_DYNAMIC_MAX="${JGPT_FP16_DYNAMIC_MAX:-65536}"

    export JGPT_FLASH_ATTENTION="${JGPT_FLASH_ATTENTION:-0}"

    export JGPT_FUSED_LM_HEAD="${JGPT_FUSED_LM_HEAD:-1}"

    export JGPT_FULL_GPU_TRAIN="${JGPT_FULL_GPU_TRAIN:-0}"
    export JGPT_GENERATE_GPU_KV="${JGPT_GENERATE_GPU_KV:-1}"
    export JGPT_GPU_E2E_TRAIN="${JGPT_GPU_E2E_TRAIN:-1}"

    export JGPT_LOG_COLOR="${JGPT_LOG_COLOR:-}"
    export JGPT_MAX_SEQUENCES="${JGPT_MAX_SEQUENCES:-}"

    export JGPT_PROFILE="${JGPT_PROFILE:-0}"
    export JGPT_PROFILE_STEPS="${JGPT_PROFILE_STEPS:-20}"
    export JGPT_TIMINGS="${JGPT_TIMINGS:-0}"
    export JGPT_TRAIN_GPU_RESIDENT="${JGPT_TRAIN_GPU_RESIDENT:-1}"
}

jgpt_e2e_train_overrides() {
    export JGPT_TRAIN_PERF="${JGPT_TRAIN_PERF:-1}"
    export JGPT_TRAIN_LOSS_MODE="${JGPT_TRAIN_LOSS_MODE:-full}"
    export JGPT_SAMPLED_CE_CANDIDATES="${JGPT_SAMPLED_CE_CANDIDATES:-128}"
    export JGPT_SAMPLED_CE_NEGATIVE_MODE="${JGPT_SAMPLED_CE_NEGATIVE_MODE:-batch_shared_uniform}"
    export JGPT_DECODER_LAYER_CUDA_GRAPH="${JGPT_DECODER_LAYER_CUDA_GRAPH:-1}"
    export JGPT_FUSED_FFN_RMS_W1W3="${JGPT_FUSED_FFN_RMS_W1W3:-1}"
    export JGPT_FP16_DYNAMIC_INITIAL="${JGPT_FP16_DYNAMIC_INITIAL:-8192}"
    export JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK="${JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK:-256}"
    export JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH="${JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH:-0}"
}

jgpt_cmake_build_cuda() {
    cmake -B build -S src/main/cpp
    cmake --build build
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
}

jgpt_resolve_mvn_command() {
    while [[ "${1:-}" == e2e ]]; do
        shift
        export JGPT_GPU_E2E_TRAIN=1
        export JGPT_FULL_GPU_TRAIN=1
    done
    # Совместимость: старый вызов «… e2e allbooks …» — слово allbooks игнорируем.
    while [[ "${1:-}" == allbooks ]]; do
        shift
    done

    local -a cmd
    if [[ $# -gt 0 ]]; then
        cmd=(
            mvn -q compile exec:java
            -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
            "-Dexec.args=$*"
        )
    else
        cmd=(
            mvn -q compile exec:java
            -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain
            '-Dexec.args=--boo .'
        )
    fi
    JGPT_EXEC_CMD=("${cmd[@]}")
}

# ─── Параметры smart-монитора ─────────────────────────────────
PRESETS=("00-max-throughput" "01-aggressive" "02-stable" "03-recovery" "04-minimal")
# Порог «стабильного прогресса»: число строк лога с улучшением best (см. EVAL_IMPROVEMENTS) для upgrade
STABLE_EVALS_FOR_UPGRADE=30
OOM_THRESHOLD=1
FP16_STUCK_THRESHOLD=8
HANG_SECONDS=300
PLATEAU_THRESHOLD=15
MONITOR_INTERVAL=30

STATE_DIR="$ROOT/state"
LOG_FILE="$ROOT/training_allbooks.log"
PID_FILE="$STATE_DIR/training.pid"
PRESET_FILE="$STATE_DIR/current_preset_idx"

mkdir -p "$STATE_DIR"

# ─── Разбор аргументов ────────────────────────────────────────
START_PRESET=""
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            echo "Использование: $0 [ПРЕСЕТ]"
            echo "Пресеты: ${PRESETS[*]}"
            exit 0
            ;;
        --*) ;;
        *) START_PRESET="$arg" ;;
    esac
done

preset_index() {
    local name="$1"
    local i
    for i in "${!PRESETS[@]}"; do
        if [[ "${PRESETS[$i]}" == "$name" ]]; then
            echo "$i"
            return 0
        fi
    done
    echo "-1"
    return 0
}

# Допустимый индекс пресета [0, n) или сообщение в stderr и fallback.
clamp_preset_idx() {
    local idx="$1"
    local max=$(( ${#PRESETS[@]} - 1 ))
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "  [SMART] [WARN] Некорректный индекс пресета «${idx}», используем 01-aggressive (1)." >&2
        echo 1
        return
    fi
    if (( idx < 0 || idx > max )); then
        echo "  [SMART] [WARN] Индекс пресета $idx вне [0,$max], используем 1." >&2
        echo 1
        return
    fi
    echo "$idx"
}

# Начальный пресет
if [[ -n "$START_PRESET" ]]; then
    CURRENT_IDX=$(preset_index "$START_PRESET")
    if [[ "$CURRENT_IDX" -lt 0 ]]; then
        echo "[ERROR] Неизвестный пресет: $START_PRESET"
        exit 1
    fi
elif [[ -f "$PRESET_FILE" ]]; then
    CURRENT_IDX=$(clamp_preset_idx "$(tr -d ' \t\r\n' < "$PRESET_FILE")")
elif [[ -L "$STATE_DIR/current.env" ]]; then
    LINKED=$(basename "$(readlink "$STATE_DIR/current.env")" .env)
    CURRENT_IDX=$(preset_index "$LINKED")
    if [[ "$CURRENT_IDX" -lt 0 ]]; then
        CURRENT_IDX=1
    fi
else
    CURRENT_IDX=1
fi
CURRENT_IDX=$(clamp_preset_idx "$CURRENT_IDX")

# ─── Вспомогательные функции smart ───────────────────────────

apply_preset() {
    local idx="$1"
    idx=$(clamp_preset_idx "$idx")
    local name="${PRESETS[$idx]}"
    local env_file="$ROOT/env/${name}.env"
    if [[ ! -f "$env_file" ]]; then
        echo "  [SMART] [ERROR] Нет файла пресета: $env_file"
        exit 1
    fi
    ln -sf "../env/${name}.env" "$STATE_DIR/current.env"
    echo "$idx" > "$PRESET_FILE"
    # source только в текущем shell, не в $(...).
    set -a
    # shellcheck source=/dev/null
    source "$env_file"
    set +a
    APPLIED_PRESET_NAME="$name"
}

count_pattern_from_line() {
    local file="$1" pattern="$2" start_line="$3"
    awk -v start="$start_line" -v pat="$pattern" \
        'NR >= start && $0 ~ pat { count++ } END { print count+0 }' "$file" 2>/dev/null || echo 0
}

# Максимум подряд eval без улучшения best_loss (в сегменте с start_line).
count_plateau_evals() {
    local file="$1" start_line="$2"
    awk -v start="$start_line" '
        NR < start { next }
        /\[EVAL\].*: loss=.*/ {
            n1 = split($0, a, /: loss=/)
            if (n1 < 2) next
            cur = a[2]; sub(/ .*/, "", cur)
            n2 = split($0, b, /лучший сохранённый=/)
            if (n2 < 2) next
            best = b[2]; sub(/[^0-9.].*/, "", best)
            if (cur == best) {
                consec = 0
            } else {
                consec++
                if (consec > max_c) max_c = consec
            }
        }
        END { print max_c+0 }
    ' "$file" 2>/dev/null || echo 0
}

# Секунды от полуночи до «сейчас» минус секунды от полуночи последней метки HH:MM:SS
# в строке [STEP] или [PERF]…шаг N в сегменте лога (с start_line). Учёт перехода через полночь — грубый.
last_step_time() {
    local start_line="${1:-1}"
    local last_line
    last_line=$(awk -v start="$start_line" '
        NR >= start && /\[STEP\]/ { line = $0 }
        NR >= start && /\[PERF\]/ && /шаг[[:space:]]+[0-9]/ { line = $0 }
        END { if (line != "") print line }
    ' "$LOG_FILE" 2>/dev/null || true)
    local stamp
    stamp=$(grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2}' <<< "$last_line" || true)
    if [[ -z "$stamp" ]]; then
        echo 0
        return
    fi
    local h m s
    IFS=: read -r h m s <<< "$stamp"
    local step_secs=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
    local nh nm ns
    IFS=: read -r nh nm ns <<< "$(date +%H:%M:%S)"
    local now_day_secs=$((10#$nh * 3600 + 10#$nm * 60 + 10#$ns))
    local diff=$((now_day_secs - step_secs))
    if (( diff < 0 )); then
        diff=$((diff + 86400))
    fi
    echo "$diff"
}

# Текущее число строк в логе (0 если файла нет).
log_file_line_count() {
    if [[ ! -f "$LOG_FILE" ]]; then
        echo 0
        return
    fi
    wc -l < "$LOG_FILE" | tr -d ' '
}

stop_training() {
    if [[ ! -f "$PID_FILE" ]]; then
        return
    fi
    local pid
    pid=$(tr -d ' \t\r\n' < "$PID_FILE" || true)
    if [[ -z "$pid" ]]; then
        rm -f "$PID_FILE"
        return
    fi
    if kill -0 "$pid" 2>/dev/null; then
        echo "  [SMART] Останавливаем обучение (PID $pid) — checkpoint будет сохранён..."
        kill -TERM "$pid" 2>/dev/null || true
        local _
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    fi
    rm -f "$PID_FILE"
}

on_smart_interrupt() {
    echo ""
    echo "  [SMART] Прерывание — останавливаем обучение..."
    stop_training
    echo "  [SMART] Готово. Resume: ./scripts/jgpt-smart.sh"
    exit 0
}

trap on_smart_interrupt INT TERM

# ─── Основной цикл ────────────────────────────────────────────

DOWNGRADE_COUNT=0
UPGRADE_COUNT=0
UPGRADE_STABLE_EVALS=0

banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " JGPT Smart Training  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Пресет    : ${PRESETS[$CURRENT_IDX]}  (idx=$CURRENT_IDX)"
    echo " Downgrade : $DOWNGRADE_COUNT  |  Upgrade : $UPGRADE_COUNT"
    echo " Лог       : $LOG_FILE"
    echo "════════════════════════════════════════════════════════════"
    echo ""
}

while true; do
    apply_preset "$CURRENT_IDX"
    PRESET_NAME="$APPLIED_PRESET_NAME"
    export JGPT_STATS_PRESET="${PRESETS[$CURRENT_IDX]}"
    export JGPT_STATS_PRESET_IDX="$CURRENT_IDX"
    UPGRADE_STABLE_EVALS=0
    banner

    smart_log_lines=$(log_file_line_count)
    LOG_START_LINE=$((smart_log_lines + 1))
    rm -f "$PID_FILE"
    # Subshell пишет BASHPID до exec — PID совпадает с mvn/java после exec.
    # JGPT_* из пресета наследуются; cmake + e2e — функциями выше.
    (
        echo "$BASHPID" > "$PID_FILE"
        jgpt__maven_opts
        jgpt__export_train_env
        jgpt_e2e_train_overrides
        jgpt_cmake_build_cuda
        jgpt_resolve_mvn_command e2e allbooks
        exec "${JGPT_EXEC_CMD[@]}"
    ) 2>&1 | tee -a "$LOG_FILE" &

    for ((smart_pid_wait = 1; smart_pid_wait <= 50; smart_pid_wait++)); do
        [[ -f "$PID_FILE" ]] && break
        sleep 0.1
    done

    if [[ ! -f "$PID_FILE" ]]; then
        echo "  [SMART] [ERROR] Не удалось получить PID обучения"
        break
    fi
    TRAIN_PID=$(tr -d ' \t\r\n' < "$PID_FILE")
    echo "  [SMART] Обучение запущено (PID=$TRAIN_PID, пресет=$PRESET_NAME)"

    STOP_REASON=""

    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "$MONITOR_INTERVAL"

        kill -0 "$TRAIN_PID" 2>/dev/null || break

        OOM_COUNT=$(count_pattern_from_line "$LOG_FILE" \
            "cudaMalloc failed|out of memory|OutOfMemoryError" "$LOG_START_LINE")
        if [[ "$OOM_COUNT" -ge "$OOM_THRESHOLD" ]]; then
            STOP_REASON="OOM ($OOM_COUNT раз)"
            break
        fi

        FP16_STUCK=$(count_pattern_from_line "$LOG_FILE" \
            "масштаб loss.*1\.000×" "$LOG_START_LINE")
        if [[ "$FP16_STUCK" -ge "$FP16_STUCK_THRESHOLD" ]]; then
            STOP_REASON="FP16 scale=1.0× залип ($FP16_STUCK шагов пропущено)"
            break
        fi

        IDLE=$(last_step_time "$LOG_START_LINE")
        if [[ "$IDLE" -gt "$HANG_SECONDS" ]] && [[ "$LOG_START_LINE" -gt 1 ]]; then
            STOP_REASON="Зависание (нет шагов $IDLE сек)"
            break
        fi

        # Счётчик строк с «лучший сохранённый=…» в сегменте (как прокси «объёма» eval-улучшений для upgrade).
        EVAL_IMPROVEMENTS=$(count_pattern_from_line "$LOG_FILE" \
            "лучший сохранённый=[0-9]" "$LOG_START_LINE")
        if [[ "$EVAL_IMPROVEMENTS" -ge 1 ]]; then
            UPGRADE_STABLE_EVALS=$EVAL_IMPROVEMENTS
        fi

        PLATEAU=$(count_plateau_evals "$LOG_FILE" "$LOG_START_LINE")
        if [[ "$PLATEAU" -ge "$PLATEAU_THRESHOLD" ]]; then
            STOP_REASON="Плато eval_loss ($PLATEAU eval подряд без улучшения)"
            break
        fi

        if [[ "$UPGRADE_STABLE_EVALS" -ge "$STABLE_EVALS_FOR_UPGRADE" ]] && [[ "$CURRENT_IDX" -gt 0 ]]; then
            STOP_REASON="UPGRADE"
            break
        fi
    done

    if [[ -n "$STOP_REASON" ]]; then
        stop_training
    fi

    EXIT_CODE=0
    wait "$TRAIN_PID" 2>/dev/null || EXIT_CODE=$?
    rm -f "$PID_FILE"

    if [[ -z "$STOP_REASON" ]]; then
        OOM_COUNT=$(count_pattern_from_line "$LOG_FILE" \
            "cudaMalloc failed|out of memory|OutOfMemoryError" "$LOG_START_LINE")
        if [[ "$OOM_COUNT" -ge "$OOM_THRESHOLD" ]]; then
            STOP_REASON="OOM (мгновенный крэш, $OOM_COUNT раз)"
        fi
    fi

    if [[ -z "$STOP_REASON" ]] && [[ "$EXIT_CODE" -eq 0 ]]; then
        echo ""
        echo "  [SMART] ✓ Обучение завершено штатно (пресет=$PRESET_NAME)"
        break
    fi

    if [[ "$STOP_REASON" == "UPGRADE" ]]; then
        NEW_IDX=$((CURRENT_IDX - 1))
        UPGRADE_COUNT=$((UPGRADE_COUNT + 1))
        echo ""
        echo "  [SMART] ↑ Upgrade #$UPGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}"
        echo "           Счётчик eval/улучшений в сегменте: $UPGRADE_STABLE_EVALS (порог: $STABLE_EVALS_FOR_UPGRADE)"
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi

    if [[ -n "$STOP_REASON" ]]; then
        echo ""
        echo "  [SMART] ⚠ Проблема обнаружена: $STOP_REASON"

        NEW_IDX=$((CURRENT_IDX + 1))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            echo "  [SMART] ✗ Достигнут последний пресет (${PRESETS[$CURRENT_IDX]})"
            echo "           Нужна ручная диагностика. Логи: $LOG_FILE"
            exit 1
        fi
        DOWNGRADE_COUNT=$((DOWNGRADE_COUNT + 1))
        echo "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}"
        CURRENT_IDX=$NEW_IDX
        UPGRADE_STABLE_EVALS=0
        sleep 3
        continue
    fi

    if [[ "$EXIT_CODE" -ne 0 ]]; then
        echo "  [SMART] ✗ Процесс завершился с кодом $EXIT_CODE"
        echo "  Последние строки лога:"
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/    /' || true
        NEW_IDX=$((CURRENT_IDX + 1))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            echo "  [SMART] Достигнут последний пресет, остановка."
            exit 1
        fi
        DOWNGRADE_COUNT=$((DOWNGRADE_COUNT + 1))
        echo "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT → ${PRESETS[$NEW_IDX]}"
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi
done

echo ""
echo "  [SMART] Downgrade: $DOWNGRADE_COUNT  |  Upgrade: $UPGRADE_COUNT"
echo "  [SMART] Финальный пресет: ${PRESETS[$CURRENT_IDX]}"
