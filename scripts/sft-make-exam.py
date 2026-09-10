#!/usr/bin/env python3
"""Build a tiny clean Russian SFT exam: capitals, arithmetic, yes/no, identity.

Writes data/sft/exam/exam.jsonl (messages[]). Repeats the unique set so packing
into seq=2048 still yields enough optimizer steps.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "data" / "sft" / "exam" / "exam.jsonl"

# (phrase after «столица», capital)
CAPITALS = [
    ("франции", "Париж"),
    ("германии", "Берлин"),
    ("италии", "Рим"),
    ("испании", "Мадрид"),
    ("великобритании", "Лондон"),
    ("англии", "Лондон"),
    ("сша", "Вашингтон"),
    ("америки", "Вашингтон"),
    ("россии", "Москва"),
    ("украины", "Киев"),
    ("беларуси", "Минск"),
    ("польши", "Варшава"),
    ("чехии", "Прага"),
    ("австрии", "Вена"),
    ("швейцарии", "Берн"),
    ("нидерландов", "Амстердам"),
    ("голландии", "Амстердам"),
    ("бельгии", "Брюссель"),
    ("швеции", "Стокгольм"),
    ("норвегии", "Осло"),
    ("финляндии", "Хельсинки"),
    ("дании", "Копенгаген"),
    ("греции", "Афины"),
    ("турции", "Анкара"),
    ("китая", "Пекин"),
    ("японии", "Токио"),
    ("южной кореи", "Сеул"),
    ("индии", "Нью-Дели"),
    ("египта", "Каир"),
    ("канады", "Оттава"),
    ("бразилии", "Бразилиа"),
    ("аргентины", "Буэнос-Айрес"),
    ("австралии", "Канберра"),
    ("мексики", "Мехико"),
    ("португалии", "Лиссабон"),
    ("венгрии", "Будапешт"),
    ("румынии", "Бухарест"),
    ("болгарии", "София"),
    ("сербии", "Белград"),
    ("хорватии", "Загреб"),
    ("казахстана", "Астана"),
    ("узбекистана", "Ташкент"),
    ("грузии", "Тбилиси"),
    ("армении", "Ереван"),
    ("азербайджана", "Баку"),
    ("латвии", "Рига"),
    ("литвы", "Вильнюс"),
    ("эстонии", "Таллин"),
    ("молдовы", "Кишинёв"),
    ("израиля", "Иерусалим"),
    ("ирана", "Тегеран"),
    ("ирака", "Багдад"),
    ("саудовской аравии", "Эр-Рияд"),
    ("оаэ", "Абу-Даби"),
    ("пакистана", "Исламабад"),
    ("индонезии", "Джакарта"),
    ("таиланда", "Бангкок"),
    ("вьетнама", "Ханой"),
    ("ирландии", "Дублин"),
    ("исландии", "Рейкьявик"),
    ("новой зеландии", "Веллингтон"),
    ("чили", "Сантьяго"),
    ("перу", "Лима"),
    ("колумбии", "Богота"),
    ("кубы", "Гавана"),
    ("юар", "Претория"),
    ("марокко", "Рабат"),
    ("нигерии", "Абуджа"),
    ("кении", "Найроби"),
    ("сингапура", "Сингапур"),
    ("малайзии", "Куала-Лумпур"),
    ("филиппин", "Манила"),
    ("монголии", "Улан-Батор"),
    ("афганистана", "Кабул"),
    ("сирии", "Дамаск"),
    ("ливана", "Бейрут"),
    ("иордании", "Амман"),
    ("кипра", "Никосия"),
    ("мальты", "Валлетта"),
    ("люксембурга", "Люксембург"),
    ("монако", "Монако"),
    ("ватикана", "Ватикан"),
    ("словакии", "Братислава"),
    ("словении", "Любляна"),
    ("албании", "Тирана"),
    ("македонии", "Скопье"),
    ("боснии", "Сараево"),
    ("черногории", "Подгорица"),
    ("кыргызстана", "Бишкек"),
    ("таджикистана", "Душанбе"),
    ("туркменистана", "Ашхабад"),
]

YES_NO = [
    ("небо голубое", "Да"),
    ("днём небо голубое", "Да"),
    ("небо зелёное", "Нет"),
    ("снег белый", "Да"),
    ("снег чёрный", "Нет"),
    ("трава зелёная", "Да"),
    ("вода мокрая", "Да"),
    ("огонь горячий", "Да"),
    ("лёд холодный", "Да"),
    ("лёд горячий", "Нет"),
    ("солнце светит днём", "Да"),
    ("ночь тёмная", "Да"),
    ("люди дышат воздухом", "Да"),
    ("рыбы живут в воде", "Да"),
    ("кошки это собаки", "Нет"),
    ("собака это кошка", "Нет"),
    ("два плюс два равно четырём", "Да"),
    ("два плюс два равно пяти", "Нет"),
    ("столица франции — париж", "Да"),
    ("столица франции — лондон", "Нет"),
    ("столица россии — москва", "Да"),
    ("столица россии — париж", "Нет"),
    ("земля круглая", "Да"),
    ("земля плоская", "Нет"),
    ("в неделе семь дней", "Да"),
    ("в неделе восемь дней", "Нет"),
    ("у человека две руки", "Да"),
    ("у человека три руки", "Нет"),
    ("молоко белое", "Да"),
    ("уголь чёрный", "Да"),
    ("сахар сладкий", "Да"),
    ("лимон кислый", "Да"),
    ("зима холодная", "Да"),
    ("лето в россии тёплое", "Да"),
    ("пингвины умеют летать", "Нет"),
    ("рыбы умеют говорить", "Нет"),
    ("вода кипит при нагревании", "Да"),
    ("лёд — это замёрзшая вода", "Да"),
]

IDENTITY_QA = [
    ("кто ты", "Я языковая модель JGPT."),
    ("кто ты?", "Я языковая модель JGPT."),
    ("что ты такое", "Я языковая модель JGPT."),
    ("что ты такое?", "Я языковая модель JGPT."),
    ("ты кто", "Я языковая модель JGPT."),
    ("ты робот?", "Да. Я языковая модель JGPT."),
    ("ты ии?", "Да. Я языковая модель JGPT."),
    ("ты искусственный интеллект?", "Да. Я языковая модель JGPT."),
    ("ты человек?", "Нет. Я языковая модель JGPT."),
    ("представься", "Я языковая модель JGPT."),
    ("представься, пожалуйста", "Я языковая модель JGPT."),
    ("как тебя зовут", "JGPT."),
    ("как тебя зовут?", "JGPT."),
    ("твоё имя", "JGPT."),
    ("привет", "Привет."),
    ("привет!", "Привет."),
    ("здравствуй", "Здравствуй."),
    ("здравствуйте", "Здравствуйте."),
    ("добрый день", "Добрый день."),
    ("добрый вечер", "Добрый вечер."),
    ("помоги мне", "Да. Задай короткий вопрос."),
    ("можешь ответить коротко?", "Да."),
]


def qa(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def capitals() -> list[dict]:
    out: list[dict] = []
    for place, capital in CAPITALS:
        p = place
        P = place[:1].upper() + place[1:]
        out.append(qa(f"столица {p}", capital + "."))
        out.append(qa(f"Столица {P}?", capital + "."))
        out.append(qa(f"Какая столица у {p}?", capital + "."))
        out.append(qa(f"Назови столицу {p}.", capital + "."))
        out.append(qa(f"Как называется столица {p}?", capital + "."))
        out.append(qa(f"Столица {P} — это?", capital + "."))
        out.append(qa(f"столица {p}", f"Столица {p} — {capital}."))
        out.append(qa(f"Какая столица у {p}?", f"Столица {p} — {capital}."))
        out.append(qa(f"{capital} — столица какой страны?", f"{capital} — столица {p}."))
        out.append(qa(f"Какой страны столица {capital}?", f"{p}."))
        out.append(qa(f"{capital} это столица?", f"Да. Столица {p}."))
    return out


def arithmetic() -> list[dict]:
    out: list[dict] = []
    for a in range(0, 13):
        for b in range(0, 13):
            s = a + b
            out.append(qa(f"сколько будет {a}+{b}", f"{a} + {b} = {s}"))
            out.append(qa(f"сколько будет {a} + {b}", f"{a} + {b} = {s}"))
            out.append(qa(f"{a} плюс {b}", f"{a} + {b} = {s}"))
            out.append(qa(f"чему равно {a}+{b}?", f"{a} + {b} = {s}"))
            out.append(qa(f"Сколько будет {a}+{b}?", str(s) + "."))
        if a >= b:
            d = a - b
            out.append(qa(f"сколько будет {a}-{b}", f"{a} - {b} = {d}"))
            out.append(qa(f"{a} минус {b}", f"{a} - {b} = {d}"))
    for a in range(0, 10):
        for b in range(0, 10):
            p = a * b
            out.append(qa(f"сколько будет {a}*{b}", f"{a} × {b} = {p}"))
            out.append(qa(f"{a} умножить на {b}", f"{a} × {b} = {p}"))
    return out


def yes_no() -> list[dict]:
    out: list[dict] = []
    for fact, ans in YES_NO:
        out.append(qa(f"ответь да или нет: {fact}", ans + "."))
        out.append(qa(f"да или нет: {fact}?", ans + "."))
        out.append(qa(f"{fact}?", ans + "."))
        out.append(qa(f"Правда ли, что {fact}?", ans + "."))
    return out


def identity() -> list[dict]:
    return [qa(u, a) for u, a in IDENTITY_QA]


def build_unique() -> list[dict]:
    rows = capitals() + arithmetic() + yes_no() + identity()
    seen: set[str] = set()
    uniq: list[dict] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    return uniq


def counts(rows: list[dict]) -> dict[str, int]:
    n = {"capital": 0, "arith": 0, "yesno": 0, "ident": 0, "other": 0}
    for row in rows:
        u = row["messages"][0]["content"].lower()
        if "столиц" in u:
            n["capital"] += 1
        elif any(x in u for x in ("сколько будет", "плюс", "минус", "умножить", "чему равно")) or any(
            ch in u for ch in "+-*"
        ):
            n["arith"] += 1
        elif "да или нет" in u or u.endswith("?") and row["messages"][1]["content"] in ("Да.", "Нет."):
            n["yesno"] += 1
        elif any(
            x in u
            for x in ("кто ты", "что ты", "ты кто", "ты робот", "ты ии", "ты человек", "представься", "зовут", "привет")
        ):
            n["ident"] += 1
        else:
            n["other"] += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Write clean Russian SFT exam JSONL")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument("--repeat", type=int, default=4, help="repeat unique set this many times")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.repeat < 1:
        print("ERROR: --repeat must be >= 1", file=sys.stderr)
        return 1

    uniq = build_unique()
    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    for _ in range(args.repeat):
        chunk = list(uniq)
        rng.shuffle(chunk)
        out_rows.extend(chunk)

    dst = args.dst if args.dst.is_absolute() else ROOT / args.dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    c = counts(uniq)
    print(
        f"unique={len(uniq)} written={len(out_rows)} repeat={args.repeat} -> {dst}"
    )
    print(
        "unique by kind: "
        + ", ".join(f"{k}={v}" for k, v in c.items() if v)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
