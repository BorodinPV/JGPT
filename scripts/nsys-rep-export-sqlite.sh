#!/usr/bin/env bash
# Экспорт готового .nsys-rep в SQLite для nsys stats / внешних запросов.
#
# Пример:
#   ./scripts/nsys-rep-export-sqlite.sh /tmp/foo.nsys-rep
#   → /tmp/foo.sqlite
#
# Дальше:
#   nsys stats /tmp/foo.sqlite --report cuda_gpu_kern_sum
#
# Замечание: у nsys export обязателен флаг --type (или --type=sqlite).
# Не разбивайте команду на строки так, чтобы отдельной «строкой стало» -t (bash попытается запустить -t).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <file.nsys-rep> [out.sqlite]" >&2
  exit 1
fi

REP="$1"
if [[ ! -f "$REP" ]]; then
  echo "Нет файла: $REP" >&2
  exit 1
fi

if [[ $# -ge 2 ]]; then
  SQL="$2"
else
  base="${REP%.nsys-rep}"
  SQL="${base}.sqlite"
fi

exec nsys export --type sqlite --force-overwrite true --output "$SQL" "$REP"
