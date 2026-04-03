#!/usr/bin/env bash
# Сборка libjgpt_cuda.so и подсказка для Maven / тестов.
#
# Пример:
#   ./scripts/build_jgpt_cuda.sh
#   export JGPT_CUDA_LIB="$PWD/build/libjgpt_cuda.so"
#   mvn test -Dtest=GptKvCacheGpuParityTest
#
# Из корня репозитория после сборки скрипт печатает готовую строку export.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build"
SO="$BUILD_DIR/libjgpt_cuda.so"

mkdir -p "$BUILD_DIR"
cmake -S "$ROOT/src/main/cpp" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "$BUILD_DIR" -j"$(nproc 2>/dev/null || echo 4)"

if [[ ! -f "$SO" ]]; then
  echo "ERROR: expected $SO" >&2
  exit 1
fi

echo "Built: $SO"
echo "Run tests from project root, e.g.:"
echo "  export JGPT_CUDA_LIB=\"$SO\""
echo "  cd \"$ROOT\" && mvn test -Dtest=GptKvCacheGpuParityTest"

# При source scripts/build_jgpt_cuda.sh — сразу export в текущую оболочку
if [[ "${BASH_SOURCE[0]:-}" != "${0}" ]]; then
  export JGPT_CUDA_LIB="$SO"
fi
