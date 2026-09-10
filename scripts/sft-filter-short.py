#!/usr/bin/env python3
"""Filter SFT JSONL to short Q&A (assistant ~<400 chars). Writes data/sft/short/*.jsonl.

Keeps the original JSON object so Java SftJsonlParser still sees the same fields
(Alpaca ``output``, not ``full_output``; skips ``bad_task`` / unrepaired ``bad_*``).

Roleplay dumps (grandmaster / gpt_roleplay) are skipped unless --include-roleplay.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "sft" / "raw"
DEFAULT_DST = ROOT / "data" / "sft" / "short"

SKIP_STEMS = {
    "grandmaster_pro_max_ru",
    "gpt_roleplay_realm_ru",
}

MANY_BLANK_LINES = re.compile(r"\n\s*\n\s*\n\s*\n")
ASSISTANT_ROLES = {"assistant", "bot", "gpt", "char", "character", "model"}
USER_ROLES = {"user", "human", "prompter"}


def _trim(s) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    return t if t else None


def _skip_file(path: Path, include_roleplay: bool) -> bool:
    stem = path.stem.lower()
    if include_roleplay:
        return False
    if stem in SKIP_STEMS:
        return True
    return "roleplay" in stem


def _turns_from_array(arr) -> tuple[list[str], list[str]]:
    users: list[str] = []
    asst: list[str] = []
    systems: list[str] = []
    pending_user: list[str] = []
    if not isinstance(arr, list):
        return users, asst
    for item in arr:
        if not isinstance(item, dict):
            continue
        role = _trim(item.get("role") or item.get("from"))
        content = _trim(item.get("content") or item.get("value"))
        if content is None:
            continue
        role_l = (role or "").lower()
        if role_l == "system":
            systems.append(content)
            continue
        if role_l in USER_ROLES:
            text = content
            if systems and not users and not asst:
                text = "\n".join(systems) + "\n" + content
                systems.clear()
            users.append(text)
            pending_user.append(text)
        elif role_l in ASSISTANT_ROLES:
            asst.append(content)
    return users, asst


def extract_users_assistants(obj: dict) -> tuple[list[str], list[str]] | None:
    lang = _trim(obj.get("answer_lang"))
    if lang and lang.lower() != "ru":
        return None

    for key in ("messages", "conversation", "conversations"):
        users, asst = _turns_from_array(obj.get(key))
        if len(users) + len(asst) >= 2 and users and asst:
            return users, asst

    inst = _trim(obj.get("instruction"))
    out = _trim(obj.get("output"))
    alt = _trim(obj.get("alternative_output"))
    label = obj.get("label")
    if label is not None:
        l = str(label).lower()
        if "bad_task" in l:
            return None
        if l.startswith("bad"):
            if alt is None:
                return None
            out = alt
    if inst is None or out is None:
        return None
    inp = _trim(obj.get("input"))
    user = inst if inp is None else inst + "\n" + inp
    return [user], [out]


def too_markdowny(text: str) -> bool:
    return text.count("###") > 2 or MANY_BLANK_LINES.search(text) is not None


def keep_example(
    users: list[str],
    asst: list[str],
    *,
    max_assistant: int,
    min_assistant: int,
    max_user: int,
) -> bool:
    if not users or not asst:
        return False
    last = asst[-1]
    if len(last) < min_assistant:
        return False
    for a in asst:
        if len(a) > max_assistant or too_markdowny(a):
            return False
    for u in users:
        if len(u) > max_user:
            return False
    return True


def filter_file(
    src: Path,
    dst: Path,
    *,
    max_assistant: int,
    min_assistant: int,
    max_user: int,
) -> tuple[int, int]:
    kept = 0
    dropped = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as inf, dst.open("w", encoding="utf-8") as out:
        for line in inf:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not isinstance(obj, dict):
                dropped += 1
                continue
            extracted = extract_users_assistants(obj)
            if extracted is None:
                dropped += 1
                continue
            users, asst = extracted
            if not keep_example(
                users,
                asst,
                max_assistant=max_assistant,
                min_assistant=min_assistant,
                max_user=max_user,
            ):
                dropped += 1
                continue
            out.write(raw + "\n")
            kept += 1
    return kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter SFT JSONL to short assistant replies")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="input dir with .jsonl")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST, help="output dir")
    ap.add_argument("--max-assistant", type=int, default=400)
    ap.add_argument("--min-assistant", type=int, default=12)
    ap.add_argument("--max-user", type=int, default=500)
    ap.add_argument(
        "--include-roleplay",
        action="store_true",
        help="keep grandmaster / roleplay files (still length-filtered)",
    )
    args = ap.parse_args()

    src = args.src if args.src.is_absolute() else ROOT / args.src
    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst
    if not src.is_dir():
        print(f"ERROR: source dir missing: {src}", file=sys.stderr)
        return 1

    files = sorted(p for p in src.rglob("*.jsonl") if p.is_file())
    if not files:
        print(f"ERROR: no .jsonl in {src}", file=sys.stderr)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    total_drop = 0
    skipped_files = 0
    for path in files:
        if _skip_file(path, args.include_roleplay):
            print(f"skip file (roleplay): {path.name}")
            skipped_files += 1
            continue
        rel = path.relative_to(src)
        out_path = dst / rel
        kept, dropped = filter_file(
            path,
            out_path,
            max_assistant=args.max_assistant,
            min_assistant=args.min_assistant,
            max_user=args.max_user,
        )
        total_kept += kept
        total_drop += dropped
        print(f"{rel}: kept={kept} dropped={dropped} -> {out_path}")
        if kept == 0 and out_path.exists():
            out_path.unlink()

    print(
        f"done: kept={total_kept} dropped={total_drop} skipped_files={skipped_files} dst={dst}"
    )
    if total_kept == 0:
        print("ERROR: no short examples kept", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
