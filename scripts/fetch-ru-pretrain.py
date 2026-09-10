#!/usr/bin/env python3
"""Download Russian Wikipedia plaintext into data/books/pretrain_txt.

Uses the official Wikimedia dump (pages-articles shard 1), not the rate-limited
random API. Optional: GitHub RusLit / Hugging Face classics via flags.
"""
from __future__ import annotations

import argparse
import bz2
import re
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "data" / "books" / "pretrain_txt"
UA = "JGPT-pretrain-fetch/1.2 (local LLM training; desktop)"

WIKI_DUMP = (
    "https://dumps.wikimedia.org/ruwiki/latest/"
    "ruwiki-latest-pages-articles1.xml-p1p224167.bz2"
)
RUSLIT_ZIP = "https://github.com/d0rj/RusLit/archive/refs/heads/main.zip"
HF_BASE = "https://huggingface.co/datasets/Imperius/ru-classic/resolve/main/per_author"
HF_AUTHORS = [
    "lermontov.txt",
    "gogol.txt",
    "dostoevsky.txt",
    "tolstoy.txt",
    "turgenev.txt",
    "chekhov.txt",
    "goncharov.txt",
    "ostrovsky.txt",
    "leskov.txt",
    "bunin.txt",
    "saltykov.txt",
]

REDIRECT = re.compile(r"^\s*#\s*redirect", re.I)
TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
LINK_PIPE = re.compile(r"\[\[[^\]|]*\|([^\]]+)\]\]")
LINK_SIMPLE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?\]\]")
TAG = re.compile(r"<[^>]+>")
REF = re.compile(r"(?is)<ref[^>]*>.*?</ref>")
BOLD_ITAL = re.compile(r"'{2,}")
HEADING = re.compile(r"^[=]{2,}\s*(.*?)\s*[=]{2,}\s*$", re.M)
FILE_LINK = re.compile(r"\[\[(?:File|Файл|Image|Изображение):[^\]]*\]\]", re.I)
CATEGORY = re.compile(r"\[\[(?:Category|Категория):[^\]]*\]\]", re.I)
TABLE = re.compile(r"(?ms)^\{\|.*?^\|\}")
WS = re.compile(r"[ \t]+\n")


def fetch_to_file(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 1.5
    last: Exception | None = None
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    for attempt in range(retries):
        try:
            downloaded = 0
            with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (32 * 1024 * 1024) < 1024 * 1024:
                        print(f"  {downloaded / (1024 * 1024):.0f} MB", file=sys.stderr)
            return
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                raise
            if e.code in (429, 500, 502, 503) and attempt + 1 < retries:
                import time

                print(f"  HTTP {e.code} — retry", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 40)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt + 1 < retries:
                import time

                print(f"  net error {e} — retry", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 40)
                continue
            raise
    raise last or RuntimeError(url)


def write_txt(dst: Path, name: str, body: str) -> int:
    safe = "".join(c if c.isalnum() or c in " ._-" else "_" for c in name)[:100].strip() or "doc"
    path = dst / f"{safe}.txt"
    n = 0
    while path.exists():
        n += 1
        path = dst / f"{safe}_{n}.txt"
    text = body if body.endswith("\n") else body + "\n"
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def wikitext_to_plain(src: str) -> str:
    text = REF.sub(" ", src)
    text = TABLE.sub(" ", text)
    text = FILE_LINK.sub(" ", text)
    text = CATEGORY.sub(" ", text)
    for _ in range(4):
        nxt = TEMPLATE.sub(" ", text)
        if nxt == text:
            break
        text = nxt
    text = LINK_PIPE.sub(r"\1", text)
    text = LINK_SIMPLE.sub(r"\1", text)
    text = TAG.sub(" ", text)
    text = BOLD_ITAL.sub("", text)
    text = HEADING.sub(r"\n\1\n", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = WS.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def extract_wiki_dump(dump: Path, dst: Path, max_articles: int, min_chars: int, max_bytes: int) -> tuple[int, int]:
    files = 0
    chars = 0
    with bz2.open(dump, "rt", encoding="utf-8", errors="replace") as fh:
        title = None
        ns = None
        for event, elem in ET.iterparse(fh, events=("end",)):
            name = local_name(elem.tag)
            if name == "title":
                title = (elem.text or "").strip()
            elif name == "ns":
                ns = (elem.text or "").strip()
            elif name == "text":
                raw = elem.text or ""
                if (
                    title
                    and ns == "0"
                    and not REDIRECT.match(raw)
                    and ":" not in title.split(" ", 1)[0]
                ):
                    plain = wikitext_to_plain(raw)
                    if len(plain) >= min_chars:
                        chars += write_txt(dst, f"wiki_{title}", title + "\n\n" + plain)
                        files += 1
                        if files % 200 == 0:
                            print(f"  wiki articles {files}, {chars / (1024 * 1024):.1f} MB text", file=sys.stderr)
                        if files >= max_articles or chars >= max_bytes:
                            elem.clear()
                            break
                title = None
                ns = None
            if name == "page":
                elem.clear()
    return files, chars


def fetch_wikipedia(dst: Path, max_articles: int, min_chars: int, max_bytes: int, keep_dump: bool) -> tuple[int, int]:
    dump_dir = ROOT / "data" / "books"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / "ruwiki-pages-articles1.xml.bz2"
    if dump_path.exists() and dump_path.stat().st_size > 10_000_000:
        print(f"using existing dump {dump_path} ({dump_path.stat().st_size / (1024 * 1024):.0f} MB)", file=sys.stderr)
    else:
        print(f"downloading Wikipedia dump:\n  {WIKI_DUMP}", file=sys.stderr)
        fetch_to_file(WIKI_DUMP, dump_path)
    print("extracting articles...", file=sys.stderr)
    files, chars = extract_wiki_dump(dump_path, dst, max_articles, min_chars, max_bytes)
    if not keep_dump:
        dump_path.unlink(missing_ok=True)
    return files, chars


def fetch_ruslit(dst: Path) -> tuple[int, int]:
    print(f"downloading RusLit zip: {RUSLIT_ZIP}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "ruslit.zip"
        fetch_to_file(RUSLIT_ZIP, zpath)
        files = 0
        chars = 0
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".txt"):
                    continue
                name = Path(info.filename).name
                if name.lower() in {"readme.txt", "license.txt"}:
                    continue
                raw = zf.read(info)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("cp1251", errors="replace")
                if len(text.strip()) < 400:
                    continue
                stem = Path(info.filename.replace("\\", "/")).stem
                parent = Path(info.filename.replace("\\", "/")).parent.name
                chars += write_txt(dst, f"ruslit_{parent}_{stem}", text)
                files += 1
        return files, chars


def fetch_hf_authors(dst: Path, authors: list[str]) -> tuple[int, int]:
    files = 0
    chars = 0
    for fname in authors:
        url = f"{HF_BASE}/{fname}"
        print(f"  hf {fname}", file=sys.stderr)
        tmp = dst / f".tmp_{fname}"
        try:
            fetch_to_file(url, tmp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  skip {fname}: 404", file=sys.stderr)
                tmp.unlink(missing_ok=True)
                continue
            raise
        text = tmp.read_text(encoding="utf-8", errors="replace")
        tmp.unlink(missing_ok=True)
        if len(text.strip()) < 1000:
            continue
        chars += write_txt(dst, f"ruclassic_{Path(fname).stem}", text)
        files += 1
    return files, chars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--source", choices=["wiki", "books", "both"], default="wiki")
    ap.add_argument("--max-articles", type=int, default=8000)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-mb", type=int, default=250, help="stop after this many MB of extracted text")
    ap.add_argument("--keep-dump", action="store_true")
    ap.add_argument("--force", action="store_true", help="download even if dst already has .txt")
    args = ap.parse_args()
    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    existing = list(dst.glob("*.txt"))
    if existing and not args.force:
        print(f"already have {len(existing)} txt in {dst} — skip fetch (pass --force to add more)")
        return 0

    total_files = 0
    total_chars = 0
    if args.source in ("wiki", "both"):
        f, c = fetch_wikipedia(
            dst,
            args.max_articles,
            args.min_chars,
            args.max_mb * 1024 * 1024,
            args.keep_dump,
        )
        print(f"wikipedia: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c
    if args.source in ("books", "both"):
        f, c = fetch_ruslit(dst)
        print(f"ruslit: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c
        f, c = fetch_hf_authors(dst, HF_AUTHORS)
        print(f"hf: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c

    print(f"wrote {total_files} files, {total_chars:,} bytes -> {dst}")
    if total_files == 0:
        print("ERROR: no text downloaded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
