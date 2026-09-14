#!/usr/bin/env python3
"""Build a small high-quality Russian SFT set: data/sft/clean/*.jsonl

Reads data/sft/raw (already downloaded IlyaGusev dumps). Drops roleplay,
refusals, URL dumps, code/markdown walls, English-heavy answers. One user
turn + one assistant turn per example. Caps per source so noisy alpaca/saiga
do not drown GPT-4 instruct.

Does not touch data/sft/short (length-only filter used by the old path).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "data" / "sft" / "raw"
DEFAULT_DST = ROOT / "data" / "sft" / "clean"

SKIP_STEMS = {
    "grandmaster_pro_max_ru",
    "gpt_roleplay_realm_ru",
    "ru_turbo_alpaca_evol_instruct",
}

# After quality gates: prefer GPT-4 instruct, then alpaca, then a little chat.
SOURCE_CAPS = {
    "ru_instruct_gpt4": 8000,
    "ru_turbo_alpaca": 6000,
    "ru_turbo_saiga": 2000,
    "oasst2_ru_main_branch": 400,
    "oasst1_ru_main_branch": 180,
    "ru_sharegpt_cleaned": 60,
}

ASSISTANT_ROLES = {"assistant", "bot", "gpt", "char", "character", "model"}
USER_ROLES = {"user", "human", "prompter"}

CYR = re.compile(r"[а-яё]", re.I)
LAT = re.compile(r"[a-z]", re.I)
WS = re.compile(r"\s+")
URL = re.compile(r"https?://|www\.|t\.me/|vk\.com/", re.I)
MD_HEAD = re.compile(r"^#{1,6}\s", re.M)
CODE_FENCE = re.compile(r"```")
NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s")
HTML = re.compile(r"</?[a-z][\w:-]*\b", re.I)
MULTI_PUNCT = re.compile(r"[!?]{4,}|\.{6,}")
CHAR_CARD = re.compile(r"ты персонаж|ты играешь роль|оставайся в образе", re.I)
META_TASK = re.compile(
    r"("
    r"напиши(те)?\s+диалог|сгенерируй(те)?\s+(продолжение\s+)?диалог|"
    r"диалект|жаргон|слэнг|"
    r"два персонажа|первый человек\s*:|второй человек\s*:"
    r")",
    re.I,
)

REFUSAL = re.compile(
    r"("
    r"я\s+не\s+могу\s+(вам\s+)?(помочь|ответить|рекомендовать|посоветовать|предоставить|"
    r"выполнить|дать\s+совет|порекомендовать)"
    r"|не\s+могу\s+(порекомендовать|рекомендовать|посоветовать|предоставить|дать\s+совет)"
    r"|как\s+(ии|искусственный\s+интеллект|языковая\s+модель|нейросеть)\b"
    r"|я\s+(всего\s+лишь|только)\s+(ии|бот|языковая\s+модель|искусственный)"
    r"|я\s+не\s+являюсь\s+(врачом|юристом|специалистом)"
    r"|openai|chatgpt|gpt-3|gpt-4|claude\.ai"
    r"|обратитесь\s+к\s+(врачу|юристу|специалисту|профессионал)"
    r"|вот\s+(несколько\s+)?(ссылок|ссылки)"
    r"|я\s+не\s+могу\s+это\s+(обсуждать|сделать)"
    r")",
    re.I,
)

SEED: list[tuple[str, str]] = [
    ("Привет", "Привет. Задай короткий вопрос."),
    ("Здравствуй", "Здравствуй."),
    ("Здравствуйте", "Здравствуйте. Чем помочь?"),
    ("Добрый день", "Добрый день."),
    ("Добрый вечер", "Добрый вечер."),
    ("Как дела?", "Хорошо. Задай вопрос — отвечу коротко."),
    ("Кто ты?", "Я языковая модель JGPT."),
    ("Что ты такое?", "Я языковая модель JGPT. Пиши коротко, я отвечу по делу."),
    ("Ты человек?", "Нет. Я языковая модель JGPT."),
    ("Как тебя зовут?", "JGPT."),
    ("Представься", "Я JGPT, небольшая языковая модель. Задай вопрос."),
    ("Отвечай коротко.", "Хорошо. Пиши вопрос."),
    ("Можешь ответить одним предложением?", "Да."),
    ("Помоги", "Да. Сформулируй вопрос."),
    ("Что такое дождь?", "Дождь — капли воды, которые падают из облаков."),
    ("Что такое снег?", "Снег — замёрзшие кристаллы воды, которые падают из облаков."),
    ("Почему небо голубое?", "Воздух рассеивает синий свет сильнее, чем красный."),
    ("Сколько дней в неделе?", "Семь."),
    ("Сколько месяцев в году?", "Двенадцать."),
    ("Сколько часов в сутках?", "Двадцать четыре."),
    ("Какой сейчас язык ты понимаешь лучше всего?", "Русский."),
    ("Переведи на русский: hello", "Привет."),
    ("Переведи на русский: thank you", "Спасибо."),
    ("Переведи на английский: спасибо", "Thank you."),
    ("Как сказать по-английски «пока»?", "Bye или goodbye."),
    ("Что тяжелее: килограмм пуха или килограмм железа?", "Одинаково: оба по килограмму."),
    ("Чем отличается чай от кофе?", "Чай заваривают из листьев чайного куста, кофе — из зёрен кофейного дерева."),
    ("Как вскипятить воду?", "Налей воду в чайник или кастрюлю и нагревай, пока не появятся пузыри."),
    ("Как сварить яйцо вкрутую?", "Положи яйцо в холодную воду, доведи до кипения и вари 8–10 минут."),
    ("Как заварить чай?", "Залей чайные листья или пакетик кипятком и подожди 2–4 минуты."),
    ("Что делать, если порезался пальцем?", "Промой водой, прижми чистой тканью. Если кровь не останавливается — к врачу."),
    ("Как часто поливать комнатные цветы?", "Когда верх земли сухой. Лучше не заливать."),
    ("Зачем нужны пароли?", "Чтобы чужой не вошёл в твою почту, банк или аккаунт."),
    ("Что такое процент?", "Сотая часть числа. 10% от 200 — это 20."),
    ("Как найти площадь прямоугольника?", "Умножь длину на ширину."),
    ("Чему равно 7+8?", "15."),
    ("Чему равно 9×6?", "54."),
    ("Столица России?", "Москва."),
    ("Столица Франции?", "Париж."),
    ("На каком материке Россия?", "Евразия."),
    ("Что такое глагол?", "Часть речи, которая обозначает действие: идти, читать, спать."),
    ("Приведи пример прилагательного.", "Красный, тёплый, высокий."),
    ("Как пишется слово «здравствуйте»?", "Через «в»: здравствуйте."),
    ("Исправь: «я щя приду»", "Я сейчас приду."),
    ("Составь короткое предложение со словом «утро».", "Утро было тихим."),
    ("Объясни слово «компромисс».", "Когда обе стороны чуть уступают, чтобы договориться."),
    ("Что такое черновик?", "Первый вариант текста, который потом правят."),
    ("Как вежливо попросить помощи?", "Скажи «подскажите, пожалуйста» и кратко опиши, что нужно."),
    ("Не выдумывай факты, если не уверен.", "Хорошо. Если не знаю — так и скажу."),
    ("Если не знаешь ответ, что скажешь?", "Не знаю точно."),
    ("Напиши отказ без канцелярита: не советуй лекарства.", "Я не врач и не подбираю лекарства."),
    ("Перескажи басню коротко: ворона и сыр.", "Ворона держала сыр, лиса похвалила её голос, ворона каркнула — сыр упал."),
    ("Дай совет, как лучше учиться словам.", "Повторяй вслух маленькими порциями каждый день, а не раз в неделю пачкой."),
    ("Как начать письмо другу?", "Напиши «привет» и сразу суть: зачем пишешь."),
    ("Чем книга отличается от статьи?", "Книга длинная и цельная, статья короче и обычно на одну тему."),
]


def _trim(s) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    return t if t else None


def _skip_file(path: Path) -> bool:
    stem = path.stem.lower()
    if stem in SKIP_STEMS:
        return True
    return "roleplay" in stem or "grandmaster" in stem


def _turns_from_array(arr) -> list[tuple[str, str]]:
    if not isinstance(arr, list):
        return []
    pending_user: str | None = None
    pairs: list[tuple[str, str]] = []
    systems: list[str] = []
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
            if systems and pending_user is None and not pairs:
                text = "\n".join(systems) + "\n" + content
                systems.clear()
            pending_user = text if pending_user is None else pending_user + "\n" + text
        elif role_l in ASSISTANT_ROLES:
            if pending_user:
                pairs.append((pending_user, content))
                pending_user = None
    return pairs


def first_pair(obj: dict) -> tuple[str, str, int] | None:
    lang = _trim(obj.get("answer_lang"))
    if lang and lang.lower() != "ru":
        return None

    for key in ("messages", "conversation", "conversations"):
        pairs = _turns_from_array(obj.get(key))
        if pairs:
            return pairs[0][0], pairs[0][1], 1

    inst = _trim(obj.get("instruction"))
    out = _trim(obj.get("output"))
    alt = _trim(obj.get("alternative_output"))
    label = obj.get("label")
    prio = 1
    if label is not None:
        l = str(label).lower()
        if "bad_task" in l:
            return None
        if l.startswith("bad"):
            out = alt
        elif l == "ok":
            prio = 2
    if inst is None or out is None:
        return None
    inp = _trim(obj.get("input"))
    user = inst if inp is None else inst + "\n" + inp
    return user, out, prio


def norm_ws(text: str) -> str:
    t = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    t = WS.sub(" ", t).strip()
    t = re.sub(r"\s+([,.!?;:])", r"\1", t)
    return t


def cyr_ratio(text: str) -> float:
    c = len(CYR.findall(text))
    l = len(LAT.findall(text))
    den = c + l
    return c / den if den else 0.0


def user_key(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\wа-яё]+", " ", t, flags=re.I)
    return WS.sub(" ", t).strip()


def reject(user: str, asst: str) -> str | None:
    if CHAR_CARD.search(user) or CHAR_CARD.search(asst):
        return "role"
    if META_TASK.search(user):
        return "meta"
    if REFUSAL.search(user) or REFUSAL.search(asst):
        return "refusal"
    if URL.search(user) or URL.search(asst):
        return "url"
    if CODE_FENCE.search(asst) or HTML.search(asst):
        return "code"
    if MD_HEAD.search(asst) or asst.count("**") >= 4 or asst.count("###") >= 1:
        return "markdown"
    if NUMBERED.search(asst) and len(NUMBERED.findall(asst)) >= 4:
        return "list"
    if MULTI_PUNCT.search(asst):
        return "punct"
    if asst.count("\n") >= 6:
        return "newlines"
    if cyr_ratio(asst) < 0.72 or cyr_ratio(user) < 0.45:
        return "lang"
    if len(CYR.findall(asst)) < 12:
        return "too_latin"
    # near-echo
    uk, ak = user_key(user), user_key(asst)
    if uk and ak and (uk == ak or (len(uk) > 20 and uk in ak and len(ak) < len(uk) + 12)):
        return "echo"
    if re.search(r"название фильма\s*$", user, re.I):
        return "placeholder"
    if re.match(r"описание сюжета", asst, re.I):
        return "rubric"
    return None


def keep_len(user: str, asst: str, *, min_a: int, max_a: int, min_u: int, max_u: int) -> bool:
    if not (min_a <= len(asst) <= max_a):
        return False
    if not (min_u <= len(user) <= max_u):
        return False
    return True


def write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for user, asst in rows:
            obj = {
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": asst},
                ]
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")


def collect_source(
    path: Path,
    *,
    min_a: int,
    max_a: int,
    min_u: int,
    max_u: int,
) -> tuple[list[tuple[int, str, str]], Counter[str]]:
    reasons: Counter[str] = Counter()
    kept: list[tuple[int, str, str]] = []
    with path.open("r", encoding="utf-8") as inf:
        for line in inf:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                reasons["json"] += 1
                continue
            if not isinstance(obj, dict):
                reasons["json"] += 1
                continue
            parsed = first_pair(obj)
            if parsed is None:
                reasons["parse"] += 1
                continue
            user, asst, prio = parsed
            user, asst = norm_ws(user), norm_ws(asst)
            if not keep_len(user, asst, min_a=min_a, max_a=max_a, min_u=min_u, max_u=max_u):
                reasons["length"] += 1
                continue
            why = reject(user, asst)
            if why:
                reasons[why] += 1
                continue
            kept.append((prio, user, asst))
            reasons["kept"] += 1
    return kept, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter SFT JSONL to a clean Russian Q&A set")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--max-assistant", type=int, default=320)
    ap.add_argument("--min-assistant", type=int, default=24)
    ap.add_argument("--max-user", type=int, default=220)
    ap.add_argument("--min-user", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-seed-dialogs", action="store_true")
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

    rng = random.Random(args.seed)
    dst.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total_written = 0
    manifest: list[str] = []

    if not args.no_seed_dialogs:
        seed_rows: list[tuple[str, str]] = []
        for u, a in SEED:
            user, asst = norm_ws(u), norm_ws(a)
            key = user_key(user)
            if not key or key in seen:
                continue
            seen.add(key)
            seed_rows.append((user, asst))
        write_jsonl(dst / "jgpt_seed.jsonl", seed_rows)
        total_written += len(seed_rows)
        print(f"jgpt_seed.jsonl: written={len(seed_rows)}")
        manifest.append(f"jgpt_seed.jsonl: written={len(seed_rows)}")

    for path in files:
        if _skip_file(path):
            print(f"skip file: {path.name}")
            continue
        rows, reasons = collect_source(
            path,
            min_a=args.min_assistant,
            max_a=args.max_assistant,
            min_u=args.min_user,
            max_u=args.max_user,
        )
        rng.shuffle(rows)
        rows.sort(key=lambda x: -x[0])
        cap = SOURCE_CAPS.get(path.stem, 400)
        uniq: list[tuple[str, str]] = []
        dups = 0
        n_ok = 0
        for prio, user, asst in rows:
            key = user_key(user)
            if not key or key in seen:
                dups += 1
                continue
            seen.add(key)
            uniq.append((user, asst))
            if prio >= 2:
                n_ok += 1
            if len(uniq) >= cap:
                break
        out_path = dst / f"{path.stem}.jsonl"
        write_jsonl(out_path, uniq)
        total_written += len(uniq)
        line = (
            f"{path.name}: in_ok={reasons['kept']} written={len(uniq)} "
            f"ok_label={n_ok} cap={cap} dups={dups} drop={dict(reasons)}"
        )
        print(line)
        manifest.append(line)
        if not uniq and out_path.exists():
            out_path.unlink()

    summary = dst / "manifest.txt"
    summary.write_text(
        "\n".join(manifest) + f"\nTOTAL={total_written}\n",
        encoding="utf-8",
    )
    print(f"done: total={total_written} dst={dst}")
    if total_written == 0:
        print("ERROR: nothing kept", file=sys.stderr)
        return 1
    if total_written < 2000:
        print("WARNING: fewer than 2000 dialogs — check filters", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
