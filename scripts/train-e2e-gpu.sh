#!/usr/bin/env bash
# Сборка libjgpt_cuda.so и запуск e2e GPU-обучения (дефолты как раньше).
# Общая логика: scripts/jgpt-gpu-train-lib.sh (тот же модуль, что и jgpt-smart.sh / run-training-gpu.sh).
#
# Использование (из корня репозитория):
#   ./scripts/train-e2e-gpu.sh
#   ./scripts/train-e2e-gpu.sh single --boo …
#   JGPT_FP16_DYNAMIC_INITIAL=8192 ./scripts/train-e2e-gpu.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=scripts/jgpt-gpu-train-lib.sh
source "$ROOT/scripts/jgpt-gpu-train-lib.sh"
jgpt__maven_opts
jgpt__export_train_env
jgpt_e2e_train_overrides
jgpt_cmake_build_cuda
jgpt_resolve_mvn_command e2e "$@"
exec "${JGPT_EXEC_CMD[@]}"
