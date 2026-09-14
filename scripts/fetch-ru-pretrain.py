#!/usr/bin/env python3
"""Download Russian Wikipedia plaintext into data/books/pretrain_txt.

Official Wikimedia article shards (not the rate-limited random API).
Optional: GitHub RusLit / Hugging Face classics via --source books|both.

Existing .txt are kept. Pass --append (or --force) to add more into a non-empty dir.
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
DUMP_DIR = ROOT / "data" / "books" / "wiki_dumps"
UA = "JGPT-pretrain-fetch/1.3 (local LLM training; desktop)"
WIKI_BASE = "https://dumps.wikimedia.org/ruwiki/latest/"

# Exact filenames from dumps.wikimedia.org/ruwiki/latest/ (Sep 2026).
WIKI_SHARDS = {
    "1": "ruwiki-latest-pages-articles1.xml-p1p224167.bz2",
    "2": "ruwiki-latest-pages-articles2.xml-p224168p1042043.bz2",
    "3": "ruwiki-latest-pages-articles3.xml-p1042044p2198269.bz2",
    "4": "ruwiki-latest-pages-articles4.xml-p2198270p3698269.bz2",
    "5": "ruwiki-latest-pages-articles5.xml-p3835773p5335772.bz2",
}

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
    "pushkin.txt",
    "gorky.txt",
    "kuprin.txt",
    "andreev.txt",
    "blok.txt",
    "esenin.txt",
    "yesenin.txt",
    "nabokov.txt",
    "bulgakov.txt",
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


def safe_stem(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in " ._-" else "_" for c in name)[:100].strip() or "doc"
    return safe


def write_txt(dst: Path, name: str, body: str, *, skip_existing: bool) -> int:
    stem = safe_stem(name)
    path = dst / f"{stem}.txt"
    if path.exists():
        if skip_existing:
            return 0
        n = 0
        while path.exists():
            n += 1
            path = dst / f"{stem}_{n}.txt"
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


PAGE_XML_MAX = 8 * 1024 * 1024


def iter_wiki_page_xml(fh):
    """Yield page XML strings; skip pages larger than PAGE_XML_MAX (avoids ET OOM)."""
    buf: list[str] = []
    inside = False
    skipping = False
    size = 0
    for line in fh:
        if not inside:
            if "<page>" in line:
                inside = True
                skipping = False
                buf = [line]
                size = len(line)
            continue
        if skipping:
            if "</page>" in line:
                inside = False
            continue
        buf.append(line)
        size += len(line)
        if size > PAGE_XML_MAX:
            skipping = True
            buf.clear()
            continue
        if "</page>" in line:
            inside = False
            yield "".join(buf)
            buf.clear()


def parse_wiki_page(page_xml: str) -> tuple[str | None, str | None, str]:
    try:
        elem = ET.fromstring(page_xml)
    except ET.ParseError:
        return None, None, ""
    title = None
    ns = None
    text = ""
    for child in elem:
        name = local_name(child.tag)
        if name == "title":
            title = (child.text or "").strip()
        elif name == "ns":
            ns = (child.text or "").strip()
        elif name == "revision":
            for rev in child:
                if local_name(rev.tag) == "text":
                    text = rev.text or ""
                    break
    return title, ns, text


def extract_wiki_dump(
    dump: Path,
    dst: Path,
    max_articles: int,
    min_chars: int,
    max_bytes: int,
    skip_existing: bool,
) -> tuple[int, int]:
    files = 0
    chars = 0
    skipped = 0
    with bz2.open(dump, "rt", encoding="utf-8", errors="replace") as fh:
        for page_xml in iter_wiki_page_xml(fh):
            title, ns, raw = parse_wiki_page(page_xml)
            if (
                not title
                or ns != "0"
                or REDIRECT.match(raw)
                or ":" in title.split(" ", 1)[0]
            ):
                continue
            plain = wikitext_to_plain(raw)
            if len(plain) < min_chars:
                continue
            n = write_txt(dst, f"wiki_{title}", title + "\n\n" + plain, skip_existing=skip_existing)
            if n == 0:
                skipped += 1
                continue
            chars += n
            files += 1
            if files % 200 == 0:
                print(
                    f"  wiki new {files}, skip {skipped}, {chars / (1024 * 1024):.1f} MB text",
                    file=sys.stderr,
                )
            if files >= max_articles or chars >= max_bytes:
                break
    if skipped:
        print(f"  skipped existing {skipped}", file=sys.stderr)
    return files, chars


def parse_shards(spec: str) -> list[str]:
    out: list[str] = []
    for part in spec.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in WIKI_SHARDS:
            raise SystemExit(f"unknown wiki shard {key!r}; known: {', '.join(WIKI_SHARDS)}")
        if key not in out:
            out.append(key)
    if not out:
        raise SystemExit("empty --wiki-shards")
    return out


def fetch_wikipedia(
    dst: Path,
    shard_ids: list[str],
    max_articles: int,
    min_chars: int,
    max_bytes: int,
    keep_dump: bool,
    skip_existing: bool,
) -> tuple[int, int]:
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    files = 0
    chars = 0
    remaining_articles = max_articles
    remaining_bytes = max_bytes
    for sid in shard_ids:
        if remaining_articles <= 0 or remaining_bytes <= 0:
            break
        fname = WIKI_SHARDS[sid]
        dump_path = DUMP_DIR / fname
        if dump_path.exists() and dump_path.stat().st_size > 10_000_000:
            print(
                f"using existing dump {dump_path.name} ({dump_path.stat().st_size / (1024 * 1024):.0f} MB)",
                file=sys.stderr,
            )
        else:
            url = WIKI_BASE + fname
            print(f"downloading Wikipedia dump {sid}:\n  {url}", file=sys.stderr)
            fetch_to_file(url, dump_path)
        print(f"extracting shard {sid}...", file=sys.stderr)
        f, c = extract_wiki_dump(
            dump_path,
            dst,
            remaining_articles,
            min_chars,
            remaining_bytes,
            skip_existing,
        )
        print(f"wikipedia shard {sid}: {f} new files, {c:,} bytes")
        files += f
        chars += c
        remaining_articles -= f
        remaining_bytes -= c
        if not keep_dump:
            dump_path.unlink(missing_ok=True)
    return files, chars


def fetch_ruslit(dst: Path, skip_existing: bool) -> tuple[int, int]:
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
                n = write_txt(dst, f"ruslit_{parent}_{stem}", text, skip_existing=skip_existing)
                if n == 0:
                    continue
                chars += n
                files += 1
        return files, chars


def fetch_hf_authors(dst: Path, authors: list[str], skip_existing: bool) -> tuple[int, int]:
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
        n = write_txt(dst, f"ruclassic_{Path(fname).stem}", text, skip_existing=skip_existing)
        if n == 0:
            continue
        chars += n
        files += 1
    return files, chars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--source", choices=["wiki", "books", "both"], default="wiki")
    ap.add_argument(
        "--wiki-shards",
        default="1",
        help="comma-separated shard ids (1-5). Example: 1,2,3",
    )
    ap.add_argument("--max-articles", type=int, default=8000, help="new wiki articles this run")
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-mb", type=int, default=250, help="stop after this many MB of NEW extracted text")
    ap.add_argument("--keep-dump", action="store_true")
    ap.add_argument("--no-keep-dump", action="store_true")
    ap.add_argument(
        "--append",
        action="store_true",
        help="add files even if dst already has .txt (skips names that exist)",
    )
    ap.add_argument("--force", action="store_true", help="same as --append")
    args = ap.parse_args()
    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)
    existing = list(dst.glob("*.txt"))
    append = args.append or args.force
    if existing and not append:
        print(f"already have {len(existing)} txt in {dst} — skip fetch (pass --append to add more)")
        return 0

    keep_dump = True if append else args.keep_dump
    if args.no_keep_dump:
        keep_dump = False
    skip_existing = True
    shards = parse_shards(args.wiki_shards)

    total_files = 0
    total_chars = 0
    if args.source in ("books", "both"):
        f, c = fetch_ruslit(dst, skip_existing)
        print(f"ruslit: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c
        f, c = fetch_hf_authors(dst, HF_AUTHORS, skip_existing)
        print(f"hf: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c
    if args.source in ("wiki", "both"):
        f, c = fetch_wikipedia(
            dst,
            shards,
            args.max_articles,
            args.min_chars,
            args.max_mb * 1024 * 1024,
            keep_dump,
            skip_existing,
        )
        print(f"wikipedia: {f} files, {c:,} bytes")
        total_files += f
        total_chars += c

    print(f"wrote {total_files} new files, {total_chars:,} bytes -> {dst}")
    n_now = len(list(dst.glob("*.txt")))
    bytes_now = sum(p.stat().st_size for p in dst.glob("*.txt"))
    print(f"corpus now: {n_now} files, {bytes_now / (1024 * 1024):.1f} MB")
    if total_files == 0 and not existing:
        print("ERROR: no text downloaded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
