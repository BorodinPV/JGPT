#!/usr/bin/env bash
# =============================================================
# jgpt-start.sh — Ручной запуск/resume с выбором пресета
#
# Использование:
#   ./scripts/jgpt-start.sh                  # авто-resume, текущий пресет
#   ./scripts/jgpt-start.sh 01-aggressive    # явный пресет
#   ./scripts/jgpt-start.sh --finetune       # сброс globalStep
#   ./scripts/jgpt-start.sh --help
#
# Для полного авто-режима с переключением пресетов:
#   ./scripts/jgpt-smart.sh
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ─── Разбор аргументов ────────────────────────────────────────
FINETUNE=0
PRESET_ARG=""
for arg in "$@"; do
    case "$arg" in
        --finetune) FINETUNE=1 ;;
        --help|-h)
            echo "Использование: $0 [ПРЕСЕТ] [--finetune]"
            echo "  ПРЕСЕТ: имя файла в env/ без .env"
            echo ""
            echo "Доступные пресеты:"
            ls env/*.env 2>/dev/null | sed 's|env/||;s|\.env||' | sed 's/^/  /'
            echo ""
            echo "Для авто-адаптации: ./scripts/jgpt-smart.sh"
            exit 0
            ;;
        --*) ;;
        *) PRESET_ARG="$arg" ;;
    esac
done

# ─── Выбор env-пресета ───────────────────────────────────────
if [[ -n "$PRESET_ARG" ]]; then
    ENV_FILE="$ROOT/env/${PRESET_ARG}.env"
    if [[ ! -f "$ENV_FILE" ]]; then
        echo "[ERROR] Пресет не найден: $ENV_FILE"
        ls env/*.env 2>/dev/null | sed 's|env/||;s|\.env||' | sed 's/^/  /'
        exit 1
    fi
    ln -sf "../env/${PRESET_ARG}.env" "$ROOT/state/current.env"
elif [[ -L "$ROOT/state/current.env" ]]; then
    ENV_FILE="$(readlink -f "$ROOT/state/current.env")"
else
    ENV_FILE="$ROOT/env/01-aggressive.env"
    ln -sf "../env/01-aggressive.env" "$ROOT/state/current.env"
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] Файл пресета не найден: $ENV_FILE"
    exit 1
fi

PRESET_NAME="$(basename "$ENV_FILE" .env)"

echo "============================================================"
echo " JGPT Training  |  $(date '+%H:%M:%S')"
echo "   Пресет  : $PRESET_NAME"

# ─── Загрузка параметров ─────────────────────────────────────
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

# ─── FINETUNE ────────────────────────────────────────────────
if [[ "$FINETUNE" -eq 1 ]]; then
    export JGPT_FINETUNE=1
    echo "   Режим   : FINETUNE (globalStep сброшен)"
else
    export JGPT_FINETUNE=0
    STEP_FILE="$ROOT/state/last_step.txt"
    if [[ -f "$ROOT/checkpoints/all_books/checkpoint_final.bin" ]]; then
        if [[ -f "$STEP_FILE" ]]; then
            echo "   Resume  : с шага $(cat "$STEP_FILE")"
        else
            echo "   Resume  : checkpoint найден"
        fi
    else
        echo "   Resume  : чекпоинт не найден (старт с нуля)"
    fi
fi

# ─── Проверка флага downgrade ────────────────────────────────
FLAG="$ROOT/state/.need_downgrade"
if [[ -f "$FLAG" ]]; then
    SUGGESTED="$(cat "$FLAG")"
    echo ""
    echo "  [WARN] Монитор рекомендует пресет: $SUGGESTED"
    echo "  Применить: ./scripts/jgpt-start.sh $SUGGESTED"
    echo "  Или авто:  ./scripts/jgpt-smart.sh"
    echo "  Сбросить:  rm -f state/.need_downgrade"
    echo ""
fi

echo "============================================================"
echo "  Лог: training_allbooks.log (append)"
echo "  Стоп: Ctrl+C (checkpoint сохранится через shutdown hook)"
echo "============================================================"
echo ""

exec "$ROOT/scripts/train-e2e-gpu.sh" allbooks 2>&1 | tee -a "$ROOT/training_allbooks.log"
