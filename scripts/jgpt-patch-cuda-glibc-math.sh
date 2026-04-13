#!/usr/bin/env bash
# Опционально: правка заголовков CUDA в /usr/local под glibc 2.41+ (noexcept у sinpi/cospi/rsqrt).
# jgpt-smart.sh обычно обходится без sudo: копия include в build/cuda_include_mirror и -I в CMake.
# См. https://forums.developer.nvidia.com/t/error-exception-specification-is-incompatible-for-cospi-sinpi-cospif-sinpif-with-glibc-2-41/323591
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NVCC="${CUDACXX:-}"
if [[ -z "$NVCC" ]]; then
    NVCC="$(command -v nvcc 2>/dev/null || true)"
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
    echo "[jgpt-patch-cuda-glibc-math] nvcc не найден. Задайте CUDACXX=/путь/к/nvcc" >&2
    exit 1
fi

NVCC_REAL="$(readlink -f "$NVCC")"
CUDA_BIN="$(dirname "$NVCC_REAL")"
MATH_H="$CUDA_BIN/../targets/x86_64-linux/include/crt/math_functions.h"
MATH_H="$(readlink -f "$MATH_H")"

if [[ ! -f "$MATH_H" ]]; then
    echo "[jgpt-patch-cuda-glibc-math] Нет файла: $MATH_H" >&2
    exit 1
fi

if grep -q 'cospi(double x) noexcept' "$MATH_H"; then
    exit 0
fi

apply_patch() {
    perl -i.bak_jgpt_glibc -pe '
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+rsqrt\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+rsqrtf\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+sinpi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+sinpif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+cospi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);/;
  s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+cospif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);/;
' "$MATH_H"
}

if apply_patch 2>/dev/null && grep -q 'cospi(double x) noexcept' "$MATH_H"; then
    rm -f "${MATH_H}.bak_jgpt_glibc"
    echo "[jgpt-patch-cuda-glibc-math] Обновлён: $MATH_H"
    exit 0
fi

echo "[jgpt-patch-cuda-glibc-math] Не удалось записать $MATH_H (нужен sudo) или шаблоны не совпали." >&2
echo "[jgpt-patch-cuda-glibc-math] Выполните один раз:" >&2
echo "  sudo env CUDACXX='$NVCC_REAL' \"$ROOT/scripts/jgpt-patch-cuda-glibc-math.sh\"" >&2
exit 1
