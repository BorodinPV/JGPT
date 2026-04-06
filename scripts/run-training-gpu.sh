#!/usr/bin/env bash
# Окружение и запуск Maven для GPU-обучения. Реализация: scripts/jgpt-gpu-train-lib.sh
#
# В логе LLMTrainer.train(): «Сводка JGPT_* (обучение)».
#
# Примеры:
#   ./scripts/train-e2e-gpu.sh          # cmake + e2e-дефолты + allbooks (или аргументы после e2e)
#   ./scripts/run-training-gpu.sh e2e
#   ./run-training.sh e2e
#   ./run-training.sh single|train [exec.args]
#   ./scripts/jgpt-smart.sh             # smart: пресеты + тот же lib внутри (без этого файла)
#
# Переопределение флагов: VAR=value ./run-training.sh ...
# Для unit-тестов с пустым env: mvn без этого скрипта.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/jgpt-gpu-train-lib.sh
source "$ROOT/scripts/jgpt-gpu-train-lib.sh"
jgpt__maven_opts
jgpt__export_train_env
jgpt_resolve_mvn_command "$@"
exec "${JGPT_EXEC_CMD[@]}"
