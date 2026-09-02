#!/usr/bin/env bash
# Собирает libjgpt_cuda.so + libjgpt_cuda_extra.so в <repo>/build (Linux).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[build-cuda] Не найден cmake. Ubuntu: sudo apt install cmake build-essential" >&2
  exit 1
fi

resolve_nvcc() {
  local p cand
  if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then
    readlink -f "${CUDACXX}" 2>/dev/null || echo "${CUDACXX}"
    return 0
  fi
  p="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "$p" && -x "$p" ]]; then
    readlink -f "$p" 2>/dev/null || echo "$p"
    return 0
  fi
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    readlink -f /usr/local/cuda/bin/nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc
    return 0
  fi
  shopt -s nullglob
  local -a vers=(/usr/local/cuda-*/bin/nvcc)
  shopt -u nullglob
  if [[ ${#vers[@]} -gt 0 ]]; then
    while IFS= read -r cand; do
      [[ -x "$cand" ]] || continue
      readlink -f "$cand" 2>/dev/null || echo "$cand"
      return 0
    done < <(printf '%s\n' "${vers[@]}" | sort -Vr)
  fi
  echo "[build-cuda] nvcc не найден. Установите CUDA Toolkit или задайте CUDACXX=/path/to/nvcc" >&2
  return 1
}

resolve_cudahostcxx() {
  local p maj
  if [[ -n "${CUDAHOSTCXX:-}" && -x "${CUDAHOSTCXX}" ]]; then
    readlink -f "${CUDAHOSTCXX}" 2>/dev/null || echo "${CUDAHOSTCXX}"
    return 0
  fi
  for p in \
    /usr/bin/g++-13 /usr/bin/x86_64-linux-gnu-g++-13 \
    /usr/bin/g++-12 /usr/bin/x86_64-linux-gnu-g++-12 \
    /usr/bin/g++-11 /usr/bin/x86_64-linux-gnu-g++-11
  do
    [[ -x "$p" ]] || continue
    readlink -f "$p" 2>/dev/null || echo "$p"
    return 0
  done
  maj="$(gcc -dumpversion 2>/dev/null | cut -d. -f1)"
  maj="${maj//[^0-9]/}"
  maj="${maj:-99}"
  if [[ "$maj" -gt 13 ]]; then
    echo "[build-cuda] GCC ${maj} не поддерживается nvcc 12.x как host compiler." >&2
    echo "[build-cuda] Установите: sudo apt install g++-13" >&2
    echo "[build-cuda] Или: export CUDAHOSTCXX=/usr/bin/g++-13" >&2
    return 1
  fi
  return 0
}

patch_cuda_math_functions_h() {
  perl -i -pe '
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+rsqrt\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 rsqrt(double x) noexcept (true);/;
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+rsqrtf\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  rsqrtf(float x) noexcept (true);/;
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+sinpi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);/;
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+sinpif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);/;
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double\s+cospi\(double x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);/;
    s/^extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float\s+cospif\(float x\);$/extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);/;
  ' "$1"
}

resolve_cuda_include_mirror() {
  local nvcc_real inc_root mirror meta new_ts sys_mf
  nvcc_real="$(readlink -f "$1")"
  inc_root="$(readlink -f "$(dirname "$nvcc_real")/../targets/x86_64-linux/include")"
  if [[ ! -d "$inc_root" ]]; then
    echo "[build-cuda] Нет каталога заголовков CUDA: $inc_root" >&2
    return 1
  fi
  sys_mf="$inc_root/crt/math_functions.h"
  if grep -q 'cospi(double x) noexcept' "$sys_mf" 2>/dev/null; then
    return 0
  fi
  mirror="$ROOT/build/cuda_include_mirror"
  meta="$mirror/.jgpt_cuda_mirror_meta"
  new_ts="$(stat -c '%Y' "$inc_root/cuda_runtime.h" 2>/dev/null || echo 0)"
  if [[ -f "$mirror/crt/math_functions.h" ]] \
      && grep -q 'cospi(double x) noexcept' "$mirror/crt/math_functions.h" 2>/dev/null \
      && [[ -f "$meta" ]] && [[ "$(<"$meta")" == "${inc_root}|${new_ts}" ]]; then
    echo "$mirror"
    return 0
  fi
  echo "[build-cuda] Копирую заголовки CUDA в build/cuda_include_mirror…" >&2
  rm -rf "$mirror"
  mkdir -p "$mirror"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$inc_root/" "$mirror/"
  else
    cp -a "$inc_root/." "$mirror/"
  fi
  patch_cuda_math_functions_h "$mirror/crt/math_functions.h" || return 1
  printf '%s|%s\n' "$inc_root" "$new_ts" > "$meta"
  echo "$mirror"
}

# Кэш CMake с другой машины/ОС (напр. C:/Users/... на Linux) ломает configure.
if [[ -f build/CMakeCache.txt ]]; then
  cache_src=""
  if grep -q '^CMAKE_HOME_DIRECTORY:INTERNAL=' build/CMakeCache.txt 2>/dev/null; then
    cache_src="$(grep -m1 '^CMAKE_HOME_DIRECTORY:INTERNAL=' build/CMakeCache.txt | cut -d= -f2-)"
  fi
  expected_src="$ROOT/src/main/cpp"
  stale=0
  if [[ -n "$cache_src" && "$cache_src" != "$expected_src" ]]; then
    stale=1
  fi
  if [[ -f build/jgpt_cuda.dll || -f build/cublas64_13.dll ]]; then
    stale=1
  fi
  if [[ "$stale" -eq 1 ]]; then
    echo "[build-cuda] Stale build cache (was: ${cache_src:-Windows artifacts}) — removing build/"
    rm -rf build
  fi
fi

nvcc_path="$(resolve_nvcc)"
export CUDACXX="$nvcc_path"
cuda_bin="$(dirname "$nvcc_path")"
case ":${PATH}:" in
  *:"${cuda_bin}":*) ;;
  *) export PATH="${cuda_bin}:${PATH}" ;;
esac

if [[ -x "$ROOT/scripts/linux/fetch-cudnn.sh" ]]; then
  bash "$ROOT/scripts/linux/fetch-cudnn.sh" || echo "[build-cuda] cuDNN optional: ${ROOT}/scripts/linux/fetch-cudnn.sh не удался"
fi

cmake_args=(
  -DCMAKE_CUDA_COMPILER="$nvcc_path"
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-native}"
)
hostcxx="$(resolve_cudahostcxx || true)"
if [[ -n "$hostcxx" ]]; then
  export CUDAHOSTCXX="$hostcxx"
  cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$hostcxx")
fi

cuda_mirror="$(resolve_cuda_include_mirror "$nvcc_path" || true)"
if [[ -n "$cuda_mirror" ]]; then
  h="$(printf '%s' "$cuda_mirror" | cksum | awk '{print $1}')"
  cuda_inc_link="/tmp/jgpt-cuda-include-${UID:-0}-${h}"
  ln -sfn "$cuda_mirror" "$cuda_inc_link"
  cmake_args+=("-DCMAKE_CUDA_FLAGS=-I${cuda_inc_link}")
fi

cmake -B build -U CMAKE_CUDA_FLAGS -S src/main/cpp "${cmake_args[@]}"
cmake --build build --parallel

SO="$ROOT/build/libjgpt_cuda.so"
if [[ ! -f "$SO" ]]; then
  echo "[build-cuda] Сборка прошла, но $SO не найден." >&2
  exit 1
fi
export JGPT_CUDA_LIB="$SO"
echo "OK: $SO"
echo "  export JGPT_CUDA_LIB=$SO"
