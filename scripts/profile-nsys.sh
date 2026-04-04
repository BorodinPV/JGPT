#!/usr/bin/env bash
# Запись таймлайна Nsight Systems (CUDA + OS) на несколько секунд обучения.
# Требуется: nsys из NVIDIA Nsight Systems (пакет nsight-systems / nsight-systems-cli).
#
# Примеры:
#   ./scripts/profile-nsys.sh
#   ./scripts/profile-nsys.sh single
#   NSYS_OUT=/tmp/my ./scripts/profile-nsys.sh
#
# Отчёт: jgpt-profile.nsys-rep в корне проекта (или $NSYS_OUT без суффикса).
#
# Не прерывайте снимок Ctrl+C: .qdstrm без финализации → нет .nsys-rep или битый поток.
# По умолчанию JGPT_EXIT_AFTER_STEP=8 — JVM выходит сама, nsys дописывает отчёт.
# Отключить авто-стоп: JGPT_EXIT_AFTER_STEP=0 ./scripts/profile-nsys.sh
#
# Если после прогона есть только *.qdstrm и nsys пишет «Importer binary … not found» —
# скрипт подставляет LD_LIBRARY_PATH для host-linux-x64 и вызывает QdstrmImporter вручную.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v nsys &>/dev/null; then
  echo "nsys не найден. Установите Nsight Systems (например пакет nsight-systems) и добавьте в PATH."
  exit 1
fi

NSIGHT_HOST=""
for c in \
  /usr/lib/nsight-systems/host-linux-x64 \
  /usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64 \
  /usr/lib/nsight-systems.BAK/host-linux-x64 \
  /usr/local/lib/nsight-systems/host-linux-x64; do
  if [[ -x "$c/QdstrmImporter" ]]; then
    NSIGHT_HOST="$c"
    break
  fi
done
if [[ -n "$NSIGHT_HOST" ]]; then
  export LD_LIBRARY_PATH="${NSIGHT_HOST}:${LD_LIBRARY_PATH:-}"
fi

OUT="${NSYS_OUT:-$ROOT/jgpt-profile}"
export JGPT_EXIT_AFTER_STEP="${JGPT_EXIT_AFTER_STEP:-8}"
export JGPT_TRAIN_PERF="${JGPT_TRAIN_PERF:-0}"

echo "Префикс вывода nsys: ${OUT}  (ожидается ${OUT}.nsys-rep)"
echo "Ранний выход: JGPT_EXIT_AFTER_STEP=${JGPT_EXIT_AFTER_STEP} (0 = без лимита шагов)"
if [[ -n "$NSIGHT_HOST" ]]; then
  echo "Nsight host libs: LD_LIBRARY_PATH+=$NSIGHT_HOST (QdstrmImporter для nsys)"
else
  echo "Предупреждение: каталог host-linux-x64 с QdstrmImporter не найден — авто-импорт .qdstrm может не сработать." >&2
fi
echo "Просмотр: nsys-ui ${OUT}.nsys-rep"
echo "SQLite:   ./scripts/nsys-rep-export-sqlite.sh ${OUT}.nsys-rep"

set +e
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  -o "$OUT" \
  "$ROOT/run-training.sh" "$@"
NSYS_RC=$?
set -e

NR="${OUT}.nsys-rep"
QD="${OUT}.qdstrm"
if [[ ! -f "$NR" && -f "$QD" ]]; then
  echo "[profile-nsys] Нет ${NR}, ручной импорт ${QD} → ${NR}"
  if [[ -n "$NSIGHT_HOST" ]]; then
    if ! "$NSIGHT_HOST/QdstrmImporter" --input-file "$QD" --output-file "$NR" --force-overwrite; then
      echo "[profile-nsys] QdstrmImporter завершился с ошибкой. Попробуйте: ./scripts/qdstrm-to-nsys-rep.sh $QD $NR" >&2
    fi
  else
    "$ROOT/scripts/qdstrm-to-nsys-rep.sh" "$QD" "$NR" || true
  fi
fi

if [[ -f "$NR" ]]; then
  echo "[profile-nsys] Отчёт: $NR"
else
  echo "[profile-nsys] Внимание: $NR не создан (rc=$NSYS_RC)." >&2
fi
exit "$NSYS_RC"
