#!/usr/bin/env python3
"""Quality stats for the books/wiki pretrain .txt corpus (no GPU, no tokenizer)."""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "data" / "books" / "pretrain_txt"

MARKUP = re.compile(
    r"(\{\{|\[\[|\]\]|==\s|Категория:|Category:|#redirect|file:|изображение:|<ref|&nbsp;|&lt;)",
    re.I,
)
CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")


def count_alpha(text: str) -> tuple[int, int]:
    cyr = lat = 0
    for ch in text:
        o = ord(ch)
        if 0x0410 <= o <= 0x044F or o in (0x0401, 0x0451):
            cyr += 1
        elif (65 <= o <= 90) or (97 <= o <= 122):
            lat += 1
    return cyr, lat


def kind(name: str) -> str:
    n = name.lower()
    if n.startswith("wiki_"):
        return "wiki"
    if n.startswith("ruslit") or "ruslit" in n:
        return "ruslit"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT)
    ap.add_argument("--preview", type=int, default=6)
    args = ap.parse_args()
    root: Path = args.data_dir
    if not root.is_dir():
        print(f"missing {root}", file=sys.stderr)
        return 1

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    files = list(root.rglob("*.txt"))
    print(f"files={len(files):,} dir={root}")

    n_ok = 0
    n_empty = 0
    n_bad_utf = 0
    n_nul = 0
    n_fffd = 0
    n_markup = 0
    n_short400 = 0
    n_short1000 = 0
    bytes_total = 0
    chars_total = 0
    cyr_total = 0
    lat_total = 0
    by_kind = Counter()
    by_kind_bytes = Counter()
    lengths: list[int] = []
    hashes: dict[bytes, list[str]] = defaultdict(list)
    hash_n: Counter[bytes] = Counter()

    for i, p in enumerate(files, 1):
        raw = p.read_bytes()
        bytes_total += len(raw)
        k = kind(p.name)
        by_kind[k] += 1
        by_kind_bytes[k] += len(raw)
        if b"\x00" in raw:
            n_nul += 1
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            n_bad_utf += 1
            text = raw.decode("utf-8", errors="replace")
        if "\ufffd" in text:
            n_fffd += 1
        if not text.strip():
            n_empty += 1
            continue
        n_ok += 1
        nch = len(text)
        chars_total += nch
        lengths.append(nch)
        if nch < 400:
            n_short400 += 1
        if nch < 1000:
            n_short1000 += 1
        if i % 20 == 1:
            cc, ll = count_alpha(text)
            cyr_total += cc
            lat_total += ll
        if MARKUP.search(text):
            n_markup += 1
        h = hashlib.sha1(raw).digest()
        hash_n[h] += 1
        if len(hashes[h]) < 3:
            hashes[h].append(p.name)
        if i % 50000 == 0:
            print(f"  scanned {i:,}/{len(files):,}", flush=True)

    lengths.sort()

    def pct(p: float) -> int:
        if not lengths:
            return 0
        idx = min(len(lengths) - 1, max(0, int(round((p / 100) * (len(lengths) - 1)))))
        return lengths[idx]

    extra = sum(c - 1 for c in hash_n.values() if c > 1)
    groups = sum(1 for c in hash_n.values() if c > 1)

    letters = cyr_total + lat_total
    cyr_share = (100.0 * cyr_total / letters) if letters else 0.0
    # ~3.8 chars/token is typical for 16k Russian BPE; report range 3.5–4.5
    est_lo = chars_total / 4.5
    est_hi = chars_total / 3.5
    est_mid = chars_total / 4.0

    print()
    print("=== size ===")
    print(f"ok_nonempty={n_ok:,} empty={n_empty:,} utf8_errors={n_bad_utf:,} nul={n_nul:,} ffdd={n_fffd:,}")
    print(f"bytes={bytes_total:,} ({bytes_total/1e6:.1f} MB) chars={chars_total:,}")
    print(f"est_tokens 16k BPE ~ {est_hi/1e9:.2f}-{est_lo/1e9:.2f}B unique (mid 4 ch/tok = {est_mid/1e9:.2f}B)")
    print(f"cyrillic_letters={cyr_share:.1f}% of [A-zА-я]  latin={100-cyr_share:.1f}%")
    print()
    print("=== by filename kind ===")
    for k, n in by_kind.most_common():
        print(f"  {k}: {n:,} files, {by_kind_bytes[k]/1e6:.1f} MB")
    others = [p.name for p in files if kind(p.name) == "other"][:12]
    if others:
        print("  other examples: " + ", ".join(others))
    print()
    print("=== length (chars) ===")
    print(
        f"min={lengths[0] if lengths else 0} p10={pct(10)} p25={pct(25)} p50={pct(50)} "
        f"p75={pct(75)} p90={pct(90)} p99={pct(99)} max={lengths[-1] if lengths else 0}"
    )
    print(f"shorter_than_400={n_short400:,}  shorter_than_1000={n_short1000:,}")
    print()
    print("=== wiki leftover markup ===")
    print(f"files_with_markup_residue={n_markup:,} ({100.0*n_markup/max(n_ok,1):.1f}%)")
    print()
    print("=== exact duplicates (sha1 of whole file) ===")
    print(f"duplicate_groups={groups:,} extra_copies={extra:,} unique_contents={len(hash_n):,}")
    for h, c in hash_n.most_common(8):
        if c < 2:
            break
        print(f"  x{c}: {', '.join(hashes[h])}")

    print()
    print("=== random previews ===")
    rng = random.Random(42)
    sample = rng.sample(files, k=min(args.preview, len(files)))
    for p in sample:
        t = p.read_text(encoding="utf-8", errors="replace").replace("\n", " ")
        print(f"-- {p.name} ({len(t)} chars)")
        print("  " + t[:220].strip().encode("utf-8", "replace").decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
