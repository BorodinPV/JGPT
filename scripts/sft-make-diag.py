#!/usr/bin/env python3
"""Diagnostic SFT sets: tiny overfit (16) and simple factual QA (~250).

Not for production. Overfit checks whether the train/infer pipeline can
memorize known prompts. The QA set is a short grounding ablation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Unique prompts we will score at T=0 after overfit.
OVERFIT: list[tuple[str, str]] = [
    ("2+2?", "4"),
    ("Столица Франции?", "Париж"),
    ("Какого цвета снег?", "Белого."),
    ("Сколько дней в неделе?", "Семь."),
    ("Что такое дождь?", "Дождь — это осадки в виде капель воды, выпадающие из облаков."),
    ("Привет", "Привет."),
    ("Столица России?", "Москва"),
    ("Сколько будет 3+5?", "8"),
    ("Какого цвета уголь?", "Чёрного."),
    ("Кто ты?", "Я языковая модель JGPT."),
    ("Что тяжелее: килограмм пуха или килограмм железа?", "Одинаково: оба по килограмму."),
    ("Сколько часов в сутках?", "Двадцать четыре."),
    ("Столица Германии?", "Берлин"),
    ("Почему небо голубое?", "Воздух рассеивает синий свет сильнее, чем красный."),
    ("Как тебя зовут?", "JGPT."),
    ("Чему равно 7+8?", "15"),
]

CAPITALS: list[tuple[str, str]] = [
    ("Франции", "Париж"),
    ("Германии", "Берлин"),
    ("Италии", "Рим"),
    ("Испании", "Мадрид"),
    ("Великобритании", "Лондон"),
    ("США", "Вашингтон"),
    ("России", "Москва"),
    ("Украины", "Киев"),
    ("Беларуси", "Минск"),
    ("Польши", "Варшава"),
    ("Чехии", "Прага"),
    ("Австрии", "Вена"),
    ("Швейцарии", "Берн"),
    ("Нидерландов", "Амстердам"),
    ("Бельгии", "Брюссель"),
    ("Швеции", "Стокгольм"),
    ("Норвегии", "Осло"),
    ("Финляндии", "Хельсинки"),
    ("Дании", "Копенгаген"),
    ("Греции", "Афины"),
    ("Турции", "Анкара"),
    ("Китая", "Пекин"),
    ("Японии", "Токио"),
    ("Индии", "Нью-Дели"),
    ("Египта", "Каир"),
    ("Канады", "Оттава"),
    ("Бразилии", "Бразилиа"),
    ("Австралии", "Канберра"),
    ("Португалии", "Лиссабон"),
    ("Казахстана", "Астана"),
    ("Грузии", "Тбилиси"),
    ("Армении", "Ереван"),
    ("Латвии", "Рига"),
    ("Литвы", "Вильнюс"),
    ("Эстонии", "Таллин"),
    ("Израиля", "Иерусалим"),
]

FACTS: list[tuple[str, str]] = [
    ("Что такое снег?", "Снег — замёрзшие кристаллы воды, которые падают из облаков."),
    ("Какого цвета молоко?", "Белого."),
    ("Какого цвета трава?", "Зелёного."),
    ("Какого цвета уголь?", "Чёрного."),
    ("Лёд — это что?", "Замёрзшая вода."),
    ("Из чего состоит вода?", "Из водорода и кислорода: H2O."),
    ("Сколько месяцев в году?", "Двенадцать."),
    ("Сколько минут в часе?", "Шестьдесят."),
    ("Сколько секунд в минуте?", "Шестьдесят."),
    ("Сколько дней в году обычно?", "365."),
    ("Сколько дней в високосном году?", "366."),
    ("Какой первый день недели в России?", "Понедельник."),
    ("Какая планета ближе всех к Солнцу?", "Меркурий."),
    ("Какая самая большая планета Солнечной системы?", "Юпитер."),
    ("На какой планете мы живём?", "Земля."),
    ("Сколько ног у паука?", "Восемь."),
    ("Сколько ног у насекомого?", "Шесть."),
    ("Сколько крыльев у птицы?", "Два."),
    ("Чем дышат рыбы?", "Жабрами."),
    ("Чем дышат люди?", "Лёгкими."),
    ("Какой газ мы вдыхаем?", "Кислород."),
    ("Какой газ растения выделяют на свету?", "Кислород."),
    ("Что тяжелее: килограмм железа или килограмм ваты?", "Одинаково: оба по килограмму."),
    ("Кипит ли вода при нагревании?", "Да, при 100 °C на уровне моря."),
    ("Замерзает ли вода при 0 °C?", "Да, при обычном давлении."),
    ("Солнце — это звезда?", "Да."),
    ("Луна — это звезда?", "Нет. Луна — спутник Земли."),
    ("Земля плоская?", "Нет. Земля шарообразная."),
    ("Пингвины умеют летать?", "Нет."),
    ("Рыбы умеют говорить?", "Нет."),
    ("Днём небо обычно какого цвета?", "Голубого."),
    ("Ночью небо какого цвета?", "Тёмного."),
    ("Огонь горячий?", "Да."),
    ("Лёд холодный?", "Да."),
    ("Сахар сладкий?", "Да."),
    ("Лимон кислый?", "Да."),
    ("Соль солёная?", "Да."),
    ("Чай делают из чего?", "Из листьев чайного куста."),
    ("Кофе делают из чего?", "Из зёрен кофейного дерева."),
    ("Как найти площадь прямоугольника?", "Умножь длину на ширину."),
    ("Что такое процент?", "Сотая часть числа."),
    ("10% от 200 — это сколько?", "20."),
    ("Столица Франции — Париж?", "Да."),
    ("Столица Франции — Мадрид?", "Нет. Столица Франции — Париж."),
    ("Столица России — Москва?", "Да."),
    ("Столица России — Париж?", "Нет. Столица России — Москва."),
    ("В неделе семь дней?", "Да."),
    ("В неделе восемь дней?", "Нет. В неделе семь дней."),
    ("Два плюс два равно четырём?", "Да."),
    ("Два плюс два равно пяти?", "Нет. Два плюс два равно четырём."),
    ("Переведи на русский: hello", "Привет."),
    ("Переведи на русский: thank you", "Спасибо."),
    ("Переведи на английский: спасибо", "Thank you."),
    ("Как сказать по-английски «пока»?", "Bye или goodbye."),
    ("Какой язык ты понимаешь лучше всего?", "Русский."),
    ("Ты человек?", "Нет. Я языковая модель JGPT."),
    ("Представься", "Я языковая модель JGPT."),
    ("Что ты такое?", "Я языковая модель JGPT. Пиши коротко, я отвечу по делу."),
    ("Здравствуй", "Здравствуй."),
    ("Добрый день", "Добрый день."),
]


def qa(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def unique(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def overfit_rows() -> list[dict]:
    return [qa(u, a) for u, a in OVERFIT]


def qa_rows() -> list[dict]:
    rows = overfit_rows()
    for place, capital in CAPITALS:
        rows.append(qa(f"Столица {place}?", capital))
        rows.append(qa(f"Какая столица у {place.lower()}?", capital + "."))
    for a, b in ((0, 0), (1, 1), (2, 2), (2, 3), (4, 5), (6, 7), (8, 9), (9, 9), (10, 2), (12, 3)):
        s = a + b
        rows.append(qa(f"Сколько будет {a}+{b}?", str(s) + "."))
        rows.append(qa(f"{a} плюс {b}", f"{a} + {b} = {s}"))
    for a, b in ((2, 2), (3, 4), (5, 5), (6, 7), (8, 9), (9, 0)):
        p = a * b
        rows.append(qa(f"Сколько будет {a}×{b}?", str(p) + "."))
    rows.extend(qa(u, a) for u, a in FACTS)
    return unique(rows)


def write_jsonl(path: Path, rows: list[dict], repeat: int, seed: int) -> None:
    rng = random.Random(seed)
    out: list[dict] = []
    for _ in range(repeat):
        chunk = list(rows)
        rng.shuffle(chunk)
        out.extend(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"unique={len(rows)} written={len(out)} repeat={repeat} -> {path}")


def write_prompts(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(u for u, _ in pairs) + "\n", encoding="utf-8")
    gold = path.with_name("overfit_gold.jsonl")
    with gold.open("w", encoding="utf-8") as f:
        for user, assistant in pairs:
            f.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
    print(f"prompts={len(pairs)} -> {path}")
    print(f"gold={len(pairs)} -> {gold}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Write diagnostic overfit + QA JSONL")
    ap.add_argument("--overfit-dst", type=Path, default=ROOT / "data" / "sft" / "diag_overfit" / "overfit.jsonl")
    ap.add_argument("--qa-dst", type=Path, default=ROOT / "data" / "sft" / "diag_qa" / "qa.jsonl")
    ap.add_argument("--overfit-repeat", type=int, default=16)
    ap.add_argument("--qa-repeat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    of_rows = overfit_rows()
    write_jsonl(args.overfit_dst, of_rows, args.overfit_repeat, args.seed)
    meta = ROOT / "data" / "sft" / "diag_eval"
    write_prompts(meta / "overfit_prompts.txt", OVERFIT)
    write_jsonl(args.qa_dst, qa_rows(), args.qa_repeat, args.seed + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
