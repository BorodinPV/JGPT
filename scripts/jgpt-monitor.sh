#!/usr/bin/env bash
# =============================================================
# jgpt-monitor.sh — Живой дашборд состояния обучения
#
# Показывает прогресс, eval loss, FP16 статистику.
# Пишет state/.need_downgrade при обнаружении проблем.
# НЕ останавливает обучение сам — для авто-остановки
# используйте jgpt-smart.sh
#
# Использование:
#   ./scripts/jgpt-monitor.sh          # передний план, Ctrl+C чтобы стоп
#   ./scripts/jgpt-monitor.sh &        # фоновый режим
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_FILE="$ROOT/training_allbooks.log"
STATE_DIR="$ROOT/state"
FLAG_FILE="$STATE_DIR/.need_downgrade"
INTERVAL=60

# ─── Порог для записи флага downgrade ────────────────────────
OOM_THRESHOLD=1
FP16_STUCK_THRESHOLD=8

echo "════════════════════════════════════════════════════════════"
echo " JGPT Monitor  |  Ctrl+C чтобы остановить"
echo " Лог: $LOG_FILE"
echo " Флаг: $FLAG_FILE"
echo "════════════════════════════════════════════════════════════"

PREV_OOM=0
PREV_FP16_STUCK=0
PREV_BEST_LOSS=""
PREV_LOG_LINES=0
LAST_PROGRESS_TIME=$(date +%s)
LAST_STEP=""

while true; do
    sleep "$INTERVAL"

    if [[ ! -f "$LOG_FILE" ]]; then
        echo "  $(date '+%H:%M:%S') | Лог не найден: $LOG_FILE"
        continue
    fi

    # ── Текущие метрики из всего лога ────────────────────────
    OOM=$(grep -c "cudaMalloc failed\|OutOfMemoryError\|out of memory" "$LOG_FILE" 2>/dev/null || echo 0)
    FP16_STUCK=$(grep -c "масштаб loss.*1\.000×" "$LOG_FILE" 2>/dev/null || echo 0)
    BEST_LOSS=$(grep -oE "лучший сохранённый=[0-9]+\.[0-9]+" "$LOG_FILE" 2>/dev/null \
                | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "-")
    CURRENT_STEP=$(grep -oE "шаг [0-9]+" "$LOG_FILE" 2>/dev/null \
                   | tail -1 | grep -oE "[0-9]+" || echo "-")
    CURRENT_EPOCH=$(grep -oE "эпоха [0-9]+/[0-9]+" "$LOG_FILE" 2>/dev/null \
                    | tail -1 | grep -oE "[0-9]+/[0-9]+" || echo "-")
    LAST_EVAL=$(grep -oE "loss=[0-9]+\.[0-9]+" "$LOG_FILE" 2>/dev/null \
                | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "-")
    TOKENS_S=$(grep "ток/с≈" "$LOG_FILE" 2>/dev/null \
               | tail -1 | grep -oE "ток/с≈[0-9]+" | grep -oE "[0-9]+" || echo "-")
    SKIPPED=$(grep -c "пропущен: переполнение" "$LOG_FILE" 2>/dev/null || echo 0)
    PRESET=$(basename "$(readlink -f "$STATE_DIR/current.env" 2>/dev/null)" .env 2>/dev/null || echo "?")

    # Обновить last_step.txt
    if [[ "$CURRENT_STEP" != "-" ]]; then
        echo "$CURRENT_STEP" > "$STATE_DIR/last_step.txt"
        if [[ "$CURRENT_STEP" != "$LAST_STEP" ]]; then
            LAST_PROGRESS_TIME=$(date +%s)
            LAST_STEP="$CURRENT_STEP"
        fi
    fi

    # ── Подсчёт изменений лога (только новые строки) ─────────
    CURRENT_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    LOG_DELTA=$(( CURRENT_LINES - PREV_LOG_LINES ))
    PREV_LOG_LINES=$CURRENT_LINES

    # ── Вывод статуса ─────────────────────────────────────────
    IDLE=$(( $(date +%s) - LAST_PROGRESS_TIME ))
    FP16_NEW=$(( FP16_STUCK - PREV_FP16_STUCK ))

    printf "  %s | пресет=%-20s эпоха=%-6s шаг=%-6s\n" \
        "$(date '+%H:%M:%S')" "$PRESET" "$CURRENT_EPOCH" "$CURRENT_STEP"
    printf "           | eval_loss=%-6s best=%-6s  ток/с=%s\n" \
        "$LAST_EVAL" "$BEST_LOSS" "${TOKENS_S:-?}"
    printf "           | FP16_пропуски=%d (новых: %d)  OOM=%d  idle=%ds\n" \
        "$FP16_STUCK" "$FP16_NEW" "$OOM" "$IDLE"

    # ── Анализ и запись флага ────────────────────────────────
    PROBLEM=""

    if [[ "$OOM" -gt "$PREV_OOM" ]]; then
        PROBLEM="OOM ($OOM раз)"
        echo "  ⚠  OOM обнаружен — рекомендуем 03-recovery"
        echo "03-recovery" > "$FLAG_FILE"
    fi

    if [[ "$FP16_STUCK" -ge "$FP16_STUCK_THRESHOLD" ]] \
       && [[ "$FP16_STUCK" -gt "$PREV_FP16_STUCK" ]] \
       && [[ -z "$PROBLEM" ]]; then
        PROBLEM="FP16 stuck ($FP16_STUCK шагов пропущено)"
        echo "  ⚠  FP16 scale=1.0× залип — рекомендуем 02-stable"
        echo "02-stable" > "$FLAG_FILE"
    fi

    if [[ "$IDLE" -gt 300 ]] && [[ "$CURRENT_STEP" != "-" ]] && [[ -z "$PROBLEM" ]]; then
        echo "  ⚠  Нет новых шагов уже ${IDLE}s — возможно зависание"
        echo "02-stable" > "$FLAG_FILE"
    fi

    # Сбросить флаг если он устарел (нет проблем долгое время)
    if [[ -z "$PROBLEM" ]] && [[ -f "$FLAG_FILE" ]] && [[ "$FP16_NEW" -eq 0 ]] && [[ "$OOM" -eq "$PREV_OOM" ]]; then
        # Не сбрасываем сами — пусть пользователь или smart сбросит
        :
    fi

    # Улучшение best loss?
    if [[ "$BEST_LOSS" != "-" ]] && [[ "$BEST_LOSS" != "$PREV_BEST_LOSS" ]]; then
        echo "  ✓  Новый лучший eval loss: $BEST_LOSS"
        PREV_BEST_LOSS="$BEST_LOSS"
    fi

    PREV_OOM=$OOM
    PREV_FP16_STUCK=$FP16_STUCK
done
