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
#   00-max-throughput → 01-aggressive → 02-stable → 03-recovery
#
# Использование:
#   ./scripts/jgpt-smart.sh                    # с текущего пресета
#   ./scripts/jgpt-smart.sh 01-aggressive      # явный стартовый пресет
#   Ctrl+C — остановить (checkpoint сохраняется через shutdown hook)
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ─── Конфигурация ─────────────────────────────────────────────
PRESETS=("00-max-throughput" "01-aggressive" "02-stable" "03-recovery")
# Сколько eval без улучшения eval_loss чтобы считать "стабильным прогрессом"
STABLE_EVALS_FOR_UPGRADE=30
# Сколько OOM-строк в новом сегменте лога = сигнал к downgrade
OOM_THRESHOLD=1
# Сколько "scale=1.000×" в новом сегменте = сигнал к downgrade
FP16_STUCK_THRESHOLD=8
# Секунд без нового [STEP] = зависание
HANG_SECONDS=300
# Сколько подряд eval без улучшения best_loss = плато → downgrade
PLATEAU_THRESHOLD=15
# Интервал проверки монитора (секунды)
MONITOR_INTERVAL=30
# ──────────────────────────────────────────────────────────────

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
            exit 0 ;;
        --*) ;;
        *) START_PRESET="$arg" ;;
    esac
done

# ─── Найти индекс пресета ──────────────────────────────────────
preset_index() {
    local name="$1"
    for i in "${!PRESETS[@]}"; do
        if [[ "${PRESETS[$i]}" == "$name" ]]; then
            echo "$i"; return
        fi
    done
    echo "-1"
}

# Начальный пресет
if [[ -n "$START_PRESET" ]]; then
    CURRENT_IDX=$(preset_index "$START_PRESET")
    if [[ "$CURRENT_IDX" -lt 0 ]]; then
        echo "[ERROR] Неизвестный пресет: $START_PRESET"
        exit 1
    fi
elif [[ -f "$PRESET_FILE" ]]; then
    CURRENT_IDX=$(cat "$PRESET_FILE")
elif [[ -L "$STATE_DIR/current.env" ]]; then
    # Попробовать угадать из symlink
    LINKED=$(basename "$(readlink "$STATE_DIR/current.env")" .env)
    CURRENT_IDX=$(preset_index "$LINKED")
    [[ "$CURRENT_IDX" -lt 0 ]] && CURRENT_IDX=1
else
    CURRENT_IDX=1  # 01-aggressive по умолчанию
fi

# ─── Вспомогательные функции ──────────────────────────────────

apply_preset() {
    local idx="$1"
    local name="${PRESETS[$idx]}"
    local env_file="$ROOT/env/${name}.env"
    ln -sf "../env/${name}.env" "$STATE_DIR/current.env"
    echo "$idx" > "$PRESET_FILE"
    # ВАЖНО: source должен выполняться в текущем shell, а не в subshell.
    # Поэтому apply_preset нельзя вызывать через $(...).
    # Имя пресета возвращаем через глобальную переменную APPLIED_PRESET_NAME.
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

# Считает максимум подряд идущих eval без улучшения best_loss в сегменте лога начиная с start_line.
# Строка улучшения: loss=X (лучший сохранённый=X) — оба числа совпадают.
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

last_step_time() {
    # Время последней строки [STEP] в секундах unix
    local last
    last=$(grep "\[STEP\]" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE '^[0-9]{2}:[0-9]{2}:[0-9]{2}' || echo "")
    if [[ -z "$last" ]]; then echo 0; return; fi
    # Конвертируем HH:MM:SS в секунды от полуночи
    local h m s
    IFS=: read -r h m s <<< "$last"
    local step_secs=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    local now_secs=$(date +%s)
    local now_hms=$(date +%H:%M:%S)
    local nh nm ns
    IFS=: read -r nh nm ns <<< "$now_hms"
    local now_day_secs=$(( 10#$nh * 3600 + 10#$nm * 60 + 10#$ns ))
    # Разница (с учётом переноса через полночь)
    local diff=$(( now_day_secs - step_secs ))
    [[ "$diff" -lt 0 ]] && diff=$(( diff + 86400 ))
    echo "$diff"
}

stop_training() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  [SMART] Останавливаем обучение (PID $pid) — checkpoint будет сохранён..."
            kill -TERM "$pid" 2>/dev/null || true
            # Ждём завершения до 30 секунд
            for _ in $(seq 1 30); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
        fi
        rm -f "$PID_FILE"
    fi
}

# Graceful exit по Ctrl+C
trap '
    echo ""
    echo "  [SMART] Прерывание — останавливаем обучение..."
    stop_training
    echo "  [SMART] Готово. Resume: ./scripts/jgpt-smart.sh"
    exit 0
' INT TERM

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
    UPGRADE_STABLE_EVALS=0
    banner

    # Запуск обучения: stdout+stderr → tee (консоль И файл)
    # Трюк: subshell пишет свой PID до exec — тот же PID сохраняется после exec.
    # Все JGPT_* уже в env текущего shell (apply_preset сделал set -a; source),
    # поэтому subshell наследует их автоматически.
    LOG_START_LINE=$(( $(wc -l < "$LOG_FILE" 2>/dev/null || echo 0) + 1 ))
    rm -f "$PID_FILE"
    (
        echo $BASHPID > "$PID_FILE"
        exec "$ROOT/scripts/train-e2e-gpu.sh" allbooks
    ) 2>&1 | tee -a "$LOG_FILE" &

    # Ждём пока subshell запишет PID (обычно < 0.5s)
    for _w in $(seq 1 50); do
        [[ -f "$PID_FILE" ]] && break
        sleep 0.1
    done

    if [[ ! -f "$PID_FILE" ]]; then
        echo "  [SMART] [ERROR] Не удалось получить PID обучения"
        break
    fi
    TRAIN_PID=$(cat "$PID_FILE")
    echo "  [SMART] Обучение запущено (PID=$TRAIN_PID, пресет=$PRESET_NAME)"

    STOP_REASON=""

    # ── Мониторинг ──────────────────────────────────────────────
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        sleep "$MONITOR_INTERVAL"

        # Обучение завершилось само по себе?
        kill -0 "$TRAIN_PID" 2>/dev/null || break

        # 1. Проверка OOM
        OOM_COUNT=$(count_pattern_from_line "$LOG_FILE" \
            "cudaMalloc failed|out of memory|OutOfMemoryError" "$LOG_START_LINE")
        if [[ "$OOM_COUNT" -ge "$OOM_THRESHOLD" ]]; then
            STOP_REASON="OOM ($OOM_COUNT раз)"
            break
        fi

        # 2. FP16 застрял на scale=1.0×
        FP16_STUCK=$(count_pattern_from_line "$LOG_FILE" \
            "масштаб loss.*1\.000×" "$LOG_START_LINE")
        if [[ "$FP16_STUCK" -ge "$FP16_STUCK_THRESHOLD" ]]; then
            STOP_REASON="FP16 scale=1.0× залип ($FP16_STUCK шагов пропущено)"
            break
        fi

        # 3. Зависание (нет новых шагов)
        IDLE=$(last_step_time)
        if [[ "$IDLE" -gt "$HANG_SECONDS" ]] && [[ "$LOG_START_LINE" -gt 1 ]]; then
            STOP_REASON="Зависание (нет шагов $IDLE сек)"
            break
        fi

        # Прогресс — считаем улучшения eval_loss в текущем сегменте
        EVAL_IMPROVEMENTS=$(count_pattern_from_line "$LOG_FILE" \
            "лучший сохранённый=[0-9]" "$LOG_START_LINE")
        if [[ "$EVAL_IMPROVEMENTS" -ge 1 ]]; then
            UPGRADE_STABLE_EVALS=$EVAL_IMPROVEMENTS
        fi

        # 4. Плато eval_loss: N подряд eval без улучшения → downgrade
        PLATEAU=$(count_plateau_evals "$LOG_FILE" "$LOG_START_LINE")
        if [[ "$PLATEAU" -ge "$PLATEAU_THRESHOLD" ]]; then
            STOP_REASON="Плато eval_loss ($PLATEAU eval подряд без улучшения)"
            break
        fi

        # 5. Upgrade: достаточно стабильных улучшений → пробуем более быстрый пресет
        if [[ "$UPGRADE_STABLE_EVALS" -ge "$STABLE_EVALS_FOR_UPGRADE" ]] && \
           [[ "$CURRENT_IDX" -gt 0 ]]; then
            STOP_REASON="UPGRADE"
            break
        fi
    done

    # Если мониторинг обнаружил проблему — останавливаем Java ДО wait,
    # иначе wait заблокирует навсегда (Java ещё жива, а stop_training вызывалась позже)
    if [[ -n "$STOP_REASON" ]]; then
        stop_training
    fi

    wait "$TRAIN_PID" 2>/dev/null || true
    EXIT_CODE=$?
    rm -f "$PID_FILE"

    # ── Решение после завершения ────────────────────────────────
    if [[ -z "$STOP_REASON" ]] && [[ "$EXIT_CODE" -eq 0 ]]; then
        echo ""
        echo "  [SMART] ✓ Обучение завершено штатно (пресет=$PRESET_NAME)"
        break
    fi

    if [[ "$STOP_REASON" == "UPGRADE" ]]; then
        NEW_IDX=$(( CURRENT_IDX - 1 ))
        UPGRADE_COUNT=$(( UPGRADE_COUNT + 1 ))
        echo ""
        echo "  [SMART] ↑ Upgrade #$UPGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}"
        echo "           Стабильных улучшений eval_loss: $UPGRADE_STABLE_EVALS (порог: $STABLE_EVALS_FOR_UPGRADE)"
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi

    if [[ -n "$STOP_REASON" ]]; then
        echo ""
        echo "  [SMART] ⚠ Проблема обнаружена: $STOP_REASON"

        # Downgrade
        NEW_IDX=$(( CURRENT_IDX + 1 ))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            echo "  [SMART] ✗ Достигнут последний пресет (${PRESETS[$CURRENT_IDX]})"
            echo "           Нужна ручная диагностика. Логи: $LOG_FILE"
            exit 1
        fi
        DOWNGRADE_COUNT=$(( DOWNGRADE_COUNT + 1 ))
        echo "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT: ${PRESETS[$CURRENT_IDX]} → ${PRESETS[$NEW_IDX]}"
        CURRENT_IDX=$NEW_IDX
        UPGRADE_STABLE_EVALS=0
        sleep 3
        continue
    fi

    # Аварийный выход JVM (exit!=0, но без явной причины)
    if [[ "$EXIT_CODE" -ne 0 ]]; then
        echo "  [SMART] ✗ Процесс завершился с кодом $EXIT_CODE"
        echo "  Последние строки лога:"
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
        NEW_IDX=$(( CURRENT_IDX + 1 ))
        if [[ "$NEW_IDX" -ge "${#PRESETS[@]}" ]]; then
            echo "  [SMART] Достигнут последний пресет, остановка."
            exit 1
        fi
        DOWNGRADE_COUNT=$(( DOWNGRADE_COUNT + 1 ))
        echo "  [SMART] ↓ Downgrade #$DOWNGRADE_COUNT → ${PRESETS[$NEW_IDX]}"
        CURRENT_IDX=$NEW_IDX
        sleep 3
        continue
    fi
done

echo ""
echo "  [SMART] Downgrade: $DOWNGRADE_COUNT  |  Upgrade: $UPGRADE_COUNT"
echo "  [SMART] Финальный пресет: ${PRESETS[$CURRENT_IDX]}"
