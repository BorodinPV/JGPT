#!/usr/bin/env bash
# Рекомендуемые проверки после рефакторинга JNI matmul (stream, sync, mutex).
#
# Использование (из корня репозитория JGPT/):
#   ./scripts/profile-matmul-refactor.sh test          # только JUnit
#   ./scripts/profile-matmul-refactor.sh nsys          # Nsight Systems + тот же тест
#   ./scripts/profile-matmul-refactor.sh jni-check     # -Xcheck:jni
#   ./scripts/profile-matmul-refactor.sh help
#
# Nsight Systems: в отчёте смотрите цепочки cudaMemcpyAsync → cuBLAS без лишних
# cudaStreamSynchronize между H2D и ядром/GEMM на том же потоке.
#
# Valgrind + CUDA: типично непереносимо; для GPU утечек/гонок смотрите cuda-memcheck /
# compute-sanitizer. Для чисто JNI на стороне JVM:
#   valgrind --leak-check=full --errors-for-leak-kinds=all \
#     env JGPT_CUDA_LIB=... \
#     java ... -cp target/test-classes:target/classes:... \
#     org.junit.platform.console.ConsoleLauncher --select-class=...
# (медленно; нужны подавления для libjvm/libcuda).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

usage() {
  sed -n '2,20p' "$0" | tail -n +1
}

MODE="${1:-help}"
case "$MODE" in
  test)
    "$ROOT/scripts/build_jgpt_cuda.sh"
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    exec mvn -q test -Dtest=MatmulGpuRefactorValidationTest
    ;;
  nsys)
    if ! command -v nsys &>/dev/null; then
      echo "nsys не найден (пакет nsight-systems)."
      exit 1
    fi
    "$ROOT/scripts/build_jgpt_cuda.sh"
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    mkdir -p "$ROOT/target"
    exec nsys profile \
      --trace=cuda,nvtx,osrt \
      -o "$ROOT/target/matmul-jni-refactor" \
      mvn -q test -Dtest=MatmulGpuRefactorValidationTest
    ;;
  jni-check)
    "$ROOT/scripts/build_jgpt_cuda.sh"
    export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
    exec mvn -q test -Dtest=MatmulGpuRefactorValidationTest -Djvm.check.jni=-Xcheck:jni
    ;;
  help | *)
    usage
    exit 0
    ;;
esac
