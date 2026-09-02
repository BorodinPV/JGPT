#!/usr/bin/env python3
"""Export parquet SFT sources to JSONL (messages[]) next to existing jsonl in data/sft/raw."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "sft" / "raw"


def emit_messages(path: Path, records):
    n = 0
    with path.open("w", encoding="utf-8") as out:
        for msgs in records:
            if not msgs or len(msgs) < 2:
                continue
            out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{path.name}: {n} conversations")


def main() -> int:
    sys.path.insert(0, "/tmp/jgpt-pyarrow")
    import pyarrow.parquet as pq

    gm_out = RAW / "grandmaster_pro_max_ru.jsonl"
    if not gm_out.exists():
        def gm_msgs():
            for name in ("grandmaster-train-00000.parquet", "grandmaster-train-00001.parquet"):
                t = pq.read_table(RAW / name, columns=["conversation", "answer_lang"])
                convs = t.column("conversation").to_pylist()
                langs = t.column("answer_lang").to_pylist()
                for conv, lang in zip(convs, langs):
                    if lang != "ru" or not conv:
                        continue
                    msgs = []
                    for m in conv:
                        if not isinstance(m, dict):
                            continue
                        role = (m.get("role") or "").lower()
                        content = (m.get("content") or "").strip()
                        if not content:
                            continue
                        if role in ("user", "human", "prompter"):
                            msgs.append({"role": "user", "content": content})
                        elif role in ("assistant", "bot", "gpt", "model"):
                            msgs.append({"role": "assistant", "content": content})
                    yield msgs

        emit_messages(gm_out, gm_msgs())
    else:
        print(f"skip existing {gm_out.name}")

    rp_out = RAW / "gpt_roleplay_realm_ru.jsonl"
    rp_parq = RAW / "gpt_roleplay_realm_ru.parquet"
    if rp_parq.exists() and not rp_out.exists():
        t = pq.read_table(rp_parq, columns=["name", "context", "greeting", "dialogues"])

        def rp_msgs():
            for name, ctx, greet, dialogues in zip(
                t.column("name").to_pylist(),
                t.column("context").to_pylist(),
                t.column("greeting").to_pylist(),
                t.column("dialogues").to_pylist(),
            ):
                header_user = f"Ты персонаж «{name or ''}»."
                if ctx:
                    header_user += " " + str(ctx).strip()
                for d in dialogues or []:
                    chat = d.get("chat") if isinstance(d, dict) else None
                    msgs = [{"role": "user", "content": header_user}]
                    if greet:
                        msgs.append({"role": "assistant", "content": str(greet).strip()})
                    for m in chat or []:
                        if not isinstance(m, dict):
                            continue
                        role = (m.get("role") or "").lower()
                        content = (m.get("content") or "").strip()
                        if not content:
                            continue
                        if role in ("user", "human"):
                            msgs.append({"role": "user", "content": content})
                        elif role in ("assistant", "bot", "char", "gpt"):
                            msgs.append({"role": "assistant", "content": content})
                    yield msgs

        emit_messages(rp_out, rp_msgs())
    elif rp_out.exists():
        print(f"skip existing {rp_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
