#!/usr/bin/env bash
# Собрать .nsys-rep из «сырого» .qdstrm (после nsys profile -o prefix).
# Требуется штатный QdstrmImporter из пакета nsight-systems.
#
# Важно: если сессию оборвали Ctrl+C, поток часто битый — импорт падает.
# Тогда перезаписать профиль: ./scripts/profile-nsys.sh (см. JGPT_EXIT_AFTER_STEP внутри).
#
# Пример:
#   ./scripts/qdstrm-to-nsys-rep.sh /tmp/jgpt_prof_good.qdstrm /tmp/jgpt_prof_good.nsys-rep
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <file.qdstrm> [out.nsys-rep]" >&2
  echo "  По умолчанию out = <basename>.nsys-rep рядом с .qdstrm" >&2
  exit 1
fi

IN="$1"
if [[ ! -f "$IN" ]]; then
  echo "Нет файла: $IN" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  OUT="$2"
else
  base="${IN%.qdstrm}"
  OUT="${base}.nsys-rep"
fi

QDIMP=""
for c in \
  /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  /usr/lib/x86_64-linux-gnu/nsight-systems/host-linux-x64/QdstrmImporter; do
  if [[ -x "$c" ]]; then
    QDIMP="$c"
    break
  fi
done

if [[ -z "$QDIMP" ]]; then
  echo "QdstrmImporter не найден. Установите nsight-systems (host-linux-x64)." >&2
  exit 1
fi

QDIMP_DIR="$(dirname "$QDIMP")"
export LD_LIBRARY_PATH="${QDIMP_DIR}:${LD_LIBRARY_PATH:-}"

echo "Импорт: $IN → $OUT (LD_LIBRARY_PATH+=${QDIMP_DIR})"
exec "$QDIMP" --input-file "$IN" --output-file "$OUT" --force-overwrite
