#!/usr/bin/env bash
# Сборка libjgpt_cuda.so и полный mvn test с JGPT_CUDA_LIB (локальная проверка с GPU).
#
# Примеры:
#   ./scripts/mvn_test_gpu.sh
#   ./scripts/mvn_test_gpu.sh -Dtest=GreedyGenerateParityTest
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/build_jgpt_cuda.sh"
export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
cd "$ROOT"
exec mvn -B test "$@"
