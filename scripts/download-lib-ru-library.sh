#!/usr/bin/env bash
set -euo pipefail

# Download Russian text files from https://artefact.lib.ru/library/
# Usage:
#   ./scripts/download-lib-ru-library.sh
#   ./scripts/download-lib-ru-library.sh --out data/books/libru --delay 0.4

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_URL="https://artefact.lib.ru/library/"
OUT_DIR="$ROOT/data/books/libru"
TMP_DIR="$ROOT/state/libru_download_tmp"
DELAY_SEC="0.5"
RETRIES="3"
TIMEOUT_SEC="30"
EXTRACT_ZIP="1"
EXTRACT_DIR="$ROOT/data/books/libru_extracted"
PREPARE_TXT="1"
TXT_OUT_DIR="$ROOT/data/books/libru_txt"
TXT_MIN_CHARS="500"
RUSSIAN_ONLY="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$ROOT/$2"
      shift 2
      ;;
    --delay)
      DELAY_SEC="$2"
      shift 2
      ;;
    --retries)
      RETRIES="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--out REL_PATH] [--delay SEC] [--retries N] [--timeout SEC] [--extract-zip 0|1] [--extract-dir REL_PATH] [--prepare-txt 0|1] [--txt-out REL_PATH] [--txt-min-chars N] [--russian-only 0|1]"
      exit 0
      ;;
    --extract-zip)
      EXTRACT_ZIP="$2"
      shift 2
      ;;
    --extract-dir)
      EXTRACT_DIR="$ROOT/$2"
      shift 2
      ;;
    --prepare-txt)
      PREPARE_TXT="$2"
      shift 2
      ;;
    --txt-out)
      TXT_OUT_DIR="$ROOT/$2"
      shift 2
      ;;
    --txt-min-chars)
      TXT_MIN_CHARS="$2"
      shift 2
      ;;
    --russian-only)
      RUSSIAN_ONLY="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUT_DIR" "$TMP_DIR"
if [[ "$EXTRACT_ZIP" == "1" ]]; then
  mkdir -p "$EXTRACT_DIR"
fi
if [[ "$PREPARE_TXT" == "1" ]]; then
  mkdir -p "$TXT_OUT_DIR"
fi

INDEX_HTML="$TMP_DIR/index.html"
ALL_LINKS="$TMP_DIR/all_links.txt"
TEXT_LINKS="$TMP_DIR/text_links.txt"
NORM_LINKS="$TMP_DIR/norm_links.txt"
FAILED_LINKS="$TMP_DIR/failed_links.txt"
SKIPPED_LINKS="$TMP_DIR/skipped_links.txt"

echo "[libru] Fetching index: $BASE_URL"
curl -fsSL --max-time "$TIMEOUT_SEC" \
  -A "Mozilla/5.0 (compatible; JGPT downloader/1.0)" \
  "$BASE_URL" > "$INDEX_HTML"

python3 - <<'PY' "$INDEX_HTML" "$ALL_LINKS" "$BASE_URL"
import sys
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

index_html, out_file, base_url = sys.argv[1], sys.argv[2], sys.argv[3]
base_host = urlparse(base_url).netloc
user_agent = "Mozilla/5.0 (compatible; JGPT downloader/1.0)"

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = None
        for k, v in attrs:
            if k.lower() == "href":
                href = v
                break
        if href:
            self.links.append(href.strip())

parser = LinkParser()
with open(index_html, "r", encoding="utf-8", errors="ignore") as f:
    parser.feed(f.read())

seen = set()
result = []
for href in parser.links:
    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
        continue
    abs_url = urljoin(base_url, href)
    if abs_url not in seen:
        seen.add(abs_url)
        result.append(abs_url)

# Crawl one level down into internal section pages.
internal_pages = []
for url in result:
    pu = urlparse(url)
    path_l = pu.path.lower()
    if pu.netloc != base_host:
        continue
    if path_l.endswith((".txt", ".fb2", ".rtf", ".zip", ".7z", ".rar", ".pdf")):
        continue
    if path_l.endswith("/") or path_l.endswith(".html") or path_l.endswith(".htm"):
        internal_pages.append(url)

for page_url in internal_pages:
    try:
        req = Request(page_url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=25) as resp:
            raw = resp.read()
        html = raw.decode("utf-8", errors="ignore")
    except Exception:
        continue

    p = LinkParser()
    p.feed(html)
    for href in p.links:
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(page_url, href)
        if abs_url not in seen:
            seen.add(abs_url)
            result.append(abs_url)

with open(out_file, "w", encoding="utf-8") as f:
    for url in result:
        f.write(url + "\n")
PY

# Only likely downloadable text files (no ripgrep dependency).
python3 - <<'PY' "$ALL_LINKS" "$TEXT_LINKS"
import re
import sys

inp, outp = sys.argv[1], sys.argv[2]
pat = re.compile(r"\.(txt|fb2|rtf|zip)(\?.*)?$", re.IGNORECASE)

with open(inp, "r", encoding="utf-8", errors="ignore") as f:
    urls = [line.strip() for line in f if line.strip()]

with open(outp, "w", encoding="utf-8") as f:
    for u in urls:
        if pat.search(u):
            f.write(u + "\n")
PY

if [[ ! -s "$TEXT_LINKS" ]]; then
  echo "[libru] No direct text links found on index page."
  echo "[libru] Saved raw links at: $ALL_LINKS"
  exit 0
fi

python3 - <<'PY' "$TEXT_LINKS" "$NORM_LINKS"
import re
import sys
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

inp, outp = sys.argv[1], sys.argv[2]
seen = set()

def normalize(url: str) -> str:
    p = urlparse(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    q.sort()
    return urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), ""))

with open(inp, "r", encoding="utf-8", errors="ignore") as f:
    urls = [line.strip() for line in f if line.strip()]

with open(outp, "w", encoding="utf-8") as f:
    for u in urls:
        n = normalize(u)
        if n in seen:
            continue
        seen.add(n)
        f.write(n + "\n")
PY

: > "$FAILED_LINKS"
: > "$SKIPPED_LINKS"

total="$(wc -l < "$NORM_LINKS" | tr -d ' ')"
ok=0
fail=0
skip=0
idx=0
unzipped=0
unzip_fail=0

echo "[libru] Found $total text-like links."

while IFS= read -r url; do
  idx=$((idx + 1))

  # Keep host+path in filename to avoid collisions.
  no_scheme="${url#http://}"
  no_scheme="${no_scheme#https://}"
  safe_name="$(printf '%s' "$no_scheme" | sed -E 's/[?#].*$//' | sed -E 's#[^A-Za-z0-9._/-]+#_#g' | sed -E 's#/#__#g')"
  target="$OUT_DIR/$safe_name"

  if [[ -f "$target" && -s "$target" ]]; then
    skip=$((skip + 1))
    echo "$url" >> "$SKIPPED_LINKS"
    continue
  fi

  echo "[libru] [$idx/$total] $url"
  if curl -fL --retry "$RETRIES" --max-time "$TIMEOUT_SEC" \
      -A "Mozilla/5.0 (compatible; JGPT downloader/1.0)" \
      "$url" -o "$target"; then
    ok=$((ok + 1))
    if [[ "$EXTRACT_ZIP" == "1" && "$target" == *.zip ]]; then
      if command -v unzip >/dev/null 2>&1; then
        extract_subdir="$EXTRACT_DIR/${safe_name%.zip}"
        mkdir -p "$extract_subdir"
        if unzip -o -q "$target" -d "$extract_subdir"; then
          unzipped=$((unzipped + 1))
        else
          unzip_fail=$((unzip_fail + 1))
          echo "[libru] unzip failed: $target"
        fi
      else
        unzip_fail=$((unzip_fail + 1))
        echo "[libru] unzip not found, skipping extraction for: $target"
      fi
    fi
  else
    fail=$((fail + 1))
    echo "$url" >> "$FAILED_LINKS"
    rm -f "$target"
  fi

  sleep "$DELAY_SEC"
done < "$NORM_LINKS"

echo "[libru] Done."
echo "[libru] Downloaded: $ok"
echo "[libru] Skipped existing: $skip"
echo "[libru] Failed: $fail"
if [[ "$EXTRACT_ZIP" == "1" ]]; then
  echo "[libru] Unzipped archives: $unzipped"
  echo "[libru] Unzip failures/skipped: $unzip_fail"
  echo "[libru] Extract dir: $EXTRACT_DIR"
fi
echo "[libru] Output dir: $OUT_DIR"
echo "[libru] Failed links: $FAILED_LINKS"

if [[ "$PREPARE_TXT" == "1" ]]; then
  SRC_FOR_TXT="$OUT_DIR"
  if [[ "$EXTRACT_ZIP" == "1" ]]; then
    SRC_FOR_TXT="$EXTRACT_DIR"
  fi

  echo "[libru] Preparing training txt from: $SRC_FOR_TXT"
  python3 - <<'PY' "$SRC_FOR_TXT" "$TXT_OUT_DIR" "$TXT_MIN_CHARS" "$RUSSIAN_ONLY"
import html
import re
import sys
import unicodedata as ud
import xml.etree.ElementTree as ET
from pathlib import Path

in_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
min_chars = int(sys.argv[3])
russian_only = sys.argv[4] == "1"

if not in_dir.exists():
    print(f"[prep] Input directory does not exist: {in_dir}")
    sys.exit(0)

text_exts = {".txt", ".text"}
xmlish_exts = {".fb2", ".xml", ".xhtml", ".html", ".htm"}
ru_word_hints = {
    "и", "в", "на", "что", "это", "как", "не", "по", "из", "к", "же",
    "он", "она", "они", "мы", "вы", "ты", "я", "с", "за", "для", "но",
}

def read_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1251", "koi8-r", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")

def collapse_spaces(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = html.unescape(s)
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2007", " ")
    s = s.replace("\u2011", "-")  # non-breaking hyphen
    # Remove invisible format characters that hurt tokenization.
    for ch in ("\u00AD", "\u200B", "\u200C", "\u200D", "\u2060", "\uFEFF", "\u200E", "\u200F",
               "\u202A", "\u202B", "\u202C", "\u202D", "\u202E"):
        s = s.replace(ch, "")
    s = "".join((c if (c in "\n\t" or ud.category(c) != "Cc") else " ") for c in s)
    s = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]", " ", s)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def xml_to_text(content: str) -> str:
    content = re.sub(r"<\?xml[^>]*\?>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<!DOCTYPE[^>]*>", "", content, flags=re.IGNORECASE)
    try:
        root = ET.fromstring(content)
        chunks = []
        for t in root.itertext():
            t = t.strip()
            if t:
                chunks.append(t)
        return collapse_spaces("\n".join(chunks))
    except ET.ParseError:
        stripped = re.sub(r"(?is)<script.*?>.*?</script>", " ", content)
        stripped = re.sub(r"(?is)<style.*?>.*?</style>", " ", stripped)
        stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
        return collapse_spaces(stripped)

def extract_declared_lang(content: str) -> str:
    m = re.search(r"<lang>\s*([A-Za-z\-]{2,10})\s*</lang>", content, flags=re.IGNORECASE)
    if not m:
        return ""
    return m.group(1).strip().lower()

def is_probably_russian(text: str, declared_lang: str) -> bool:
    if declared_lang.startswith("ru"):
        return True
    if declared_lang and not declared_lang.startswith("ru"):
        return False

    sample = text[:200000]
    cyr = len(re.findall(r"[А-Яа-яЁё]", sample))
    lat = len(re.findall(r"[A-Za-z]", sample))
    letters = cyr + lat
    if letters == 0:
        return False

    cyr_ratio = cyr / letters
    words = re.findall(r"[А-Яа-яЁёA-Za-z]+", sample.lower())
    if not words:
        return False
    ru_hits = sum(1 for w in words if w in ru_word_hints)
    ru_hint_ratio = ru_hits / len(words)

    # Keep conservative threshold to avoid dropping valid RU books.
    return cyr_ratio >= 0.55 or (cyr_ratio >= 0.35 and ru_hint_ratio >= 0.01)

def safe_name(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root).with_suffix(""))
    rel = rel.replace("/", "__")
    rel = re.sub(r"[^A-Za-z0-9._\-А-Яа-яЁё]+", "_", rel)
    if not rel:
        rel = path.stem
    return rel + ".txt"

files = [p for p in in_dir.rglob("*") if p.is_file()]
written = 0
skipped_ext = 0
skipped_short = 0
skipped_non_ru = 0
errors = 0

for path in files:
    ext = path.suffix.lower()
    if ext not in text_exts and ext not in xmlish_exts:
        skipped_ext += 1
        continue
    try:
        src = read_best_effort(path)
        declared_lang = extract_declared_lang(src) if ext in xmlish_exts else ""
        out_text = xml_to_text(src) if ext in xmlish_exts else collapse_spaces(src)
        if len(out_text) < min_chars:
            skipped_short += 1
            continue
        if russian_only and not is_probably_russian(out_text, declared_lang):
            skipped_non_ru += 1
            continue
        out_path = out_dir / safe_name(path, in_dir)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        written += 1
    except Exception:
        errors += 1

print(f"[prep] Source files scanned: {len(files)}")
print(f"[prep] Written txt files: {written}")
print(f"[prep] Skipped by extension: {skipped_ext}")
print(f"[prep] Skipped too short (<{min_chars} chars): {skipped_short}")
print(f"[prep] Skipped non-Russian: {skipped_non_ru}")
print(f"[prep] Errors: {errors}")
print(f"[prep] Output dir: {out_dir}")
PY
fi
