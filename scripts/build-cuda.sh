#!/usr/bin/env bash
# Собирает libjgpt_cuda.so + libjgpt_cuda_extra.so в <repo>/build (Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v cmake >/dev/null 2>&1; then
  echo "Не найден cmake. Ubuntu: sudo apt install cmake build-essential" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
  echo "Не найден nvcc. Установите CUDA Toolkit и добавьте его bin в PATH." >&2
  exit 1
fi

NVCC="$(command -v nvcc 2>/dev/null || true)"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
  export PATH="/usr/local/cuda/bin:$PATH"
fi

cmake -B build -S src/main/cpp \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
cmake --build build --parallel

SO="$ROOT/build/libjgpt_cuda.so"
if [[ ! -f "$SO" ]]; then
  echo "Сборка прошла, но $SO не найден." >&2
  exit 1
fi
export JGPT_CUDA_LIB="$SO"
echo "OK: $SO"
echo "  export JGPT_CUDA_LIB=$SO"
echo "  ./scripts/jgpt-smart.sh"
