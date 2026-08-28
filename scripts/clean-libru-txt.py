#!/usr/bin/env python3
"""Clean lib.ru training txt: FB2 metadata, footnotes, publisher junk.

Reads UTF-8 .txt produced by scripts/download-lib-ru-library.sh --prepare-txt
and rewrites them in place (original FB2 still in data/books/libru_extracted).
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata as ud
from pathlib import Path

MIN_CHARS_DEFAULT = 500

GENRE_LINE = re.compile(
    r"^(?:antique|hronoopera|prose_[a-z0-9_]+|sf_[a-z0-9_]+|adv_[a-z0-9_]+|"
    r"love_[a-z0-9_]+|det(?:ective)?_[a-z0-9_]+|child_[a-z0-9_]+|poetry_[a-z0-9_]+|"
    r"humor_[a-z0-9_]+|religion_[a-z0-9_]+|sci_[a-z0-9_]+|comp_[a-z0-9_]+|"
    r"ref_[a-z0-9_]+|nonf_[a-z0-9_]+|home_[a-z0-9_]+|tech_[a-z0-9_]+|"
    r"foreign_[a-z0-9_]+|design|computers)$",
    re.IGNORECASE,
)
LANG_LINE = re.compile(r"^(?:ru|en|uk|de|fr|es|it|pl|la|lat|ger|eng|fra|spa|ita)$", re.I)
UUID_LINE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)
DATE_LINE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$")
VERSION_LINE = re.compile(r"^(?:v\s*)?\d+\.\d+(?:\.\d+)?$", re.I)
DIGITS_LINE = re.compile(r"^\d{1,6}$")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
BACKNOTE_MARK = re.compile(r"←\s*\d{1,4}")
BRACKET_NOTE = re.compile(r"\[\s*(?:←\s*)?\d{1,4}\s*\]")
BRACE_NOTE = re.compile(r"\{\s*\d{1,4}\s*\}")
LANG_TAG_LINE = re.compile(
    r"^\(?\s*(?:лат|англ|нем|фр|итал|исп|пол|укр|греч|др\.-греч|др\.?\s*греч)\.?\s*\)?\.?$",
    re.I,
)
NOTE_SECTION = re.compile(r"^(?:notes|примечани[ея]|сноски|комментарии)\s*\.?\s*$", re.I)
META_PHRASE = re.compile(
    r"(?i)calibre|fictionbook|ooofbtools|alreader|exporttofb|finereader|"
    r"текст предоставлен|правообладател|"
    r"библиотека\s*[«\"']?\s*артефакт|isbn|удк|ббк|chernov|litmir|lib\.ru|мошков|"
    r"создание fb2|оцифрован"
)
PUBLISHER_LINE = re.compile(
    r"^(?:аст|эксмо|москва|санкт-петербург|спб|м\.|издательство)\s*\.?$",
    re.I,
)
CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")


def cyr_ratio(s: str) -> float:
    c = len(CYR.findall(s))
    l = len(LAT.findall(s))
    t = c + l
    return c / t if t else 0.0


def is_header_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if GENRE_LINE.match(s) or LANG_LINE.match(s) or UUID_LINE.match(s):
        return True
    if DATE_LINE.match(s) or VERSION_LINE.match(s) or DIGITS_LINE.match(s):
        return True
    if EMAIL_RE.search(s) or URL_RE.search(s):
        return True
    if META_PHRASE.search(s) or PUBLISHER_LINE.match(s):
        return True
    if s.startswith("©") or s.lower().startswith("copyright"):
        return True
    if re.match(r"(?i)^(печатается с разрешения|перевод\.)", s):
        return True
    if s.lower().startswith("isbn"):
        return True
    # Split FB2 name/title fragments: one or two capitalized tokens, no sentence.
    words = s.split()
    if len(s) <= 48 and len(words) <= 4 and not re.search(r"[.!?…]", s):
        if cyr_ratio(s) >= 0.5 or (LAT.search(s) and not CYR.search(s) and len(s) < 40):
            return True
    return False


def is_prose_start(line: str) -> bool:
    s = line.strip()
    if len(s) < 70:
        return False
    if is_header_line(s) and len(s) < 120:
        return False
    return cyr_ratio(s) >= 0.35


def classify_line(line: str) -> str:
    if is_header_line(line):
        return "H"
    if is_prose_start(line):
        return "P"
    return "S"


def strip_header(lines: list[str]) -> tuple[list[str], int]:
    """Drop FB2 title-info / document-info dumped before (and after) the annotation."""
    n = len(lines)
    window = min(120, n)
    classes = [classify_line(ln) for ln in lines[:window]]
    try:
        p1 = classes.index("P")
    except ValueError:
        i = 0
        while i < window and classes[i] in ("H", "S"):
            i += 1
        return lines[i:], i

    p2 = None
    for j in range(p1 + 1, window):
        if classes[j] != "P":
            continue
        if any(classes[k] == "H" for k in range(p1 + 1, j)):
            p2 = j
        break

    if p2 is not None:
        skipped = p2 - 1  # dropped 0..p1-1 and p1+1..p2-1
        return [lines[p1]] + lines[p2:], skipped
    return lines[p1:], p1


def strip_notes_tail(lines: list[str]) -> tuple[list[str], int]:
    if not lines:
        return lines, 0
    from_idx = None
    # Cut from the last "Notes / Примечания" in the second half of the file.
    half = max(0, int(len(lines) * 0.4))
    for i in range(len(lines) - 1, half - 1, -1):
        if NOTE_SECTION.match(lines[i].strip()):
            from_idx = i
            break
    # Or a run of back-reference markers near the end.
    if from_idx is None:
        for i in range(len(lines) - 1, max(half, len(lines) - 80), -1):
            s = lines[i].strip()
            if s in ("[", "]") or BACKNOTE_MARK.fullmatch(s) or re.fullmatch(r"←", s):
                # walk up to block start
                j = i
                while j > half and (
                    not lines[j].strip()
                    or lines[j].strip() in ("[", "]")
                    or BACKNOTE_MARK.search(lines[j])
                    or NOTE_SECTION.match(lines[j].strip())
                    or DIGITS_LINE.match(lines[j].strip())
                    or LANG_TAG_LINE.match(lines[j].strip())
                ):
                    if NOTE_SECTION.match(lines[j].strip()) or BACKNOTE_MARK.search(lines[j]):
                        from_idx = j
                    j -= 1
                break
    if from_idx is None:
        return lines, 0
    dropped = len(lines) - from_idx
    return lines[:from_idx], dropped


def drop_converter_lines(lines: list[str]) -> tuple[list[str], int]:
    """Drop FB2 converter / rights lines wherever they survived."""
    conv = re.compile(
        r"(?i)fictionbook|calibre|ooofbtools|alreader|exporttofb|finereader|"
        r"текст предоставлен правообладател|создание fb2|оцифрован"
    )
    out: list[str] = []
    dropped = 0
    for line in lines:
        if conv.search(line) and (cyr_ratio(line) < 0.85 or len(line.strip()) < 280):
            dropped += 1
            continue
        out.append(line)
    return out, dropped


def drop_footnote_lines(lines: list[str]) -> tuple[list[str], int]:
    out: list[str] = []
    dropped = 0
    for i, line in enumerate(lines):
        s = line.strip()
        prev = lines[i - 1].strip() if i else ""
        if not s:
            out.append(line)
            continue
        if s in ("[", "]") or BACKNOTE_MARK.fullmatch(s) or s in ("←",):
            dropped += 1
            continue
        if LANG_TAG_LINE.match(s):
            dropped += 1
            continue
        if re.match(r"(?i)^прим\.\s*(автора|публикатора|переводчика|ред)", s):
            dropped += 1
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if DIGITS_LINE.match(s) and (
            prev.endswith(("»", '"', "“", "”"))
            or LANG_TAG_LINE.match(nxt)
            or BACKNOTE_MARK.search(nxt)
            or nxt in ("[", "]")
        ):
            dropped += 1
            continue
        out.append(line)
    return out, dropped


def scrub_inline(text: str) -> str:
    text = BRACKET_NOTE.sub("", text)
    text = BRACE_NOTE.sub("", text)
    text = BACKNOTE_MARK.sub("", text)
    text = URL_RE.sub("", text)
    text = EMAIL_RE.sub("", text)
    text = UUID_LINE.sub("", text)
    text = re.sub(
        r"\{?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}?",
        "",
        text,
    )
    text = text.replace("\ufffd", "")
    text = text.replace("\u00ad", "")
    text = re.sub(r"\[\s*\]", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text


def collapse(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join((c if (c in "\n\t" or ud.category(c) != "Cc") else " ") for c in text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> tuple[str, dict[str, int]]:
    stats = {"header_lines": 0, "note_lines": 0, "fn_lines": 0, "conv_lines": 0}
    lines = text.splitlines()
    lines, stats["header_lines"] = strip_header(lines)
    lines, stats["note_lines"] = strip_notes_tail(lines)
    lines, conv_n = drop_converter_lines(lines)
    stats["conv_lines"] = conv_n
    lines, stats["fn_lines"] = drop_footnote_lines(lines)
    body = scrub_inline("\n".join(lines))
    body = collapse(body)
    return body, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        default="data/books/libru_txt",
        help="Directory with .txt files",
    )
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"No directory: {root}", file=sys.stderr)
        return 1

    files = sorted(root.glob("*.txt"))
    rewritten = 0
    deleted = 0
    unchanged = 0
    errors = 0
    chars_before = 0
    chars_after = 0
    tot = {"header_lines": 0, "note_lines": 0, "fn_lines": 0, "conv_lines": 0}

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[err] {path.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        chars_before += len(raw)
        cleaned, st = clean_text(raw)
        for k, v in st.items():
            tot[k] += v
        if len(cleaned) < args.min_chars:
            chars_after += 0
            deleted += 1
            print(f"[clean] too short, drop: {path.name} ({len(cleaned)} chars)")
            if not args.dry_run:
                path.unlink()
            continue
        chars_after += len(cleaned)
        if cleaned == raw.rstrip("\n"):
            unchanged += 1
            continue
        rewritten += 1
        if not args.dry_run:
            path.write_text(cleaned + "\n", encoding="utf-8")

    print(f"[clean] files scanned: {len(files)}")
    print(f"[clean] rewritten: {rewritten}")
    print(f"[clean] unchanged: {unchanged}")
    print(f"[clean] deleted (too short): {deleted}")
    print(f"[clean] errors: {errors}")
    print(f"[clean] dropped header lines: {tot['header_lines']}")
    print(f"[clean] dropped notes-tail lines: {tot['note_lines']}")
    print(f"[clean] dropped footnote lines: {tot['fn_lines']}")
    print(f"[clean] dropped converter lines: {tot['conv_lines']}")
    print(f"[clean] chars {chars_before:,} -> {chars_after:,}")
    if args.dry_run:
        print("[clean] dry-run: nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
