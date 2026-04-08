#!/usr/bin/env bash
# Запуск интерактивного чата с весами после AllBooksTrain (см. InferChat).
# Пример: ./scripts/jgpt-chat.sh --boo . --layers 12 --seq-len 1024
# Промпт с пробелами: ./scripts/jgpt-chat.sh --prompt 'один два три'
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${MAVEN_OPTS:-}" != *enable-native-access* ]]; then
  export MAVEN_OPTS="--enable-native-access=ALL-UNNAMED ${MAVEN_OPTS:-}"
fi
if [[ -z "${JGPT_CUDA_LIB:-}" && -f "$ROOT/build/libjgpt_cuda.so" ]]; then
  export JGPT_CUDA_LIB="$ROOT/build/libjgpt_cuda.so"
fi

# Не использовать mvn -Dexec.args="$*": exec-maven-plugin режет строку по пробелам,
# поэтому --prompt 'Привет, как дела?' превращался в лишние «неизвестные» аргументы.
mvn -q compile
CP_FILE="$(mktemp)"
trap 'rm -f "$CP_FILE"' EXIT
mvn -q dependency:build-classpath -DincludeScope=runtime -Dmdep.outputFile="$CP_FILE"
CP="$ROOT/target/classes:$(tr -d '\r\n' < "$CP_FILE")"
exec java ${MAVEN_OPTS:-} \
  --sun-misc-unsafe-memory-access=allow \
  --add-modules=jdk.incubator.vector \
  --enable-preview \
  -cp "$CP" \
  com.veles.llm.jgpt.app.InferChat "$@"
