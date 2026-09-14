#!/usr/bin/env python3
"""Fixed factual/paraphrase eval set (bench v1).

Categories:
  A — exact/simple wording
  B — paraphrase of the same fact
  C — nearby/contrast fact or trap (not the A/B target)

Frozen. Do not edit items after the 14k SFT baseline is scored.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "sft" / "diag_eval"
GOLD = EVAL / "bench_v1.jsonl"
PROMPTS = EVAL / "bench_v1_prompts.txt"


def item(iid: str, cat: str, family: str, user: str, gold: str, needles: list[str]) -> dict:
    return {
        "id": iid,
        "cat": cat,
        "family": family,
        "user": user,
        "gold": gold,
        "needles": needles,
    }


ITEMS: list[dict] = [
    # --- capitals ---
    item("fr_a", "A", "fr", "Столица Франции?", "Париж", ["париж"]),
    item("fr_b1", "B", "fr", "Назови столицу Франции", "Париж", ["париж"]),
    item("fr_b2", "B", "fr", "Какой город является столицей Франции?", "Париж", ["париж"]),
    item("fr_b3", "B", "fr", "Во Франции столицей является какой город?", "Париж", ["париж"]),
    item("ru_a", "A", "ru", "Столица России?", "Москва", ["москв"]),
    item("ru_b1", "B", "ru", "Назови столицу России", "Москва", ["москв"]),
    item("ru_b2", "B", "ru", "Какой город столица России?", "Москва", ["москв"]),
    item("de_a", "A", "de", "Столица Германии?", "Берлин", ["берлин"]),
    item("de_b1", "B", "de", "Назови столицу Германии", "Берлин", ["берлин"]),
    item("de_b2", "B", "de", "Какая столица у Германии?", "Берлин", ["берлин"]),
    item("es_a", "A", "es", "Столица Испании?", "Мадрид", ["мадрид"]),
    item("es_b1", "B", "es", "Назови столицу Испании", "Мадрид", ["мадрид"]),
    item("it_a", "A", "it", "Столица Италии?", "Рим", ["рим"]),
    item("it_b1", "B", "it", "Какой город столица Италии?", "Рим", ["рим"]),
    item("gb_a", "A", "gb", "Столица Великобритании?", "Лондон", ["лондон"]),
    item("gb_b1", "B", "gb", "Назови столицу Англии", "Лондон", ["лондон"]),
    item("cn_a", "A", "cn", "Столица Китая?", "Пекин", ["пекин"]),
    item("cn_b1", "B", "cn", "Какая столица у Китая?", "Пекин", ["пекин"]),
    item("jp_a", "A", "jp", "Столица Японии?", "Токио", ["токио"]),
    item("jp_b1", "B", "jp", "Назови столицу Японии", "Токио", ["токио"]),
    item("us_a", "A", "us", "Столица США?", "Вашингтон", ["вашингтон"]),
    item("us_b1", "B", "us", "Какой город является столицей США?", "Вашингтон", ["вашингтон"]),
    # C: nearby capitals / traps
    item("c_fr_london", "C", "trap", "Столица Франции — Лондон?", "Нет. Столица Франции — Париж.", ["нет"]),
    item("c_fr_madrid", "C", "trap", "Столица Франции — Мадрид?", "Нет. Столица Франции — Париж.", ["нет"]),
    item("c_ru_paris", "C", "trap", "Столица России — Париж?", "Нет. Столица России — Москва.", ["нет"]),
    item("c_pl", "C", "near", "Столица Польши?", "Варшава", ["варшав"]),
    item("c_ua", "C", "near", "Столица Украины?", "Киев", ["киев", "київ"]),
    item("c_by", "C", "near", "Столица Беларуси?", "Минск", ["минск"]),
    item("c_tr", "C", "near", "Столица Турции?", "Анкара", ["анкар"]),
    item("c_eg", "C", "near", "Столица Египта?", "Каир", ["каир"]),
    item("c_ca", "C", "near", "Столица Канады?", "Оттава", ["оттав"]),
    item("c_br", "C", "near", "Столица Бразилии?", "Бразилиа", ["бразили"]),
    # --- arithmetic ---
    item("add_22_a", "A", "add22", "2+2?", "4", ["4"]),
    item("add_22_b1", "B", "add22", "Сколько будет два плюс два?", "4", ["4"]),
    item("add_22_b2", "B", "add22", "Чему равно 2 плюс 2?", "4", ["4"]),
    item("add_35_a", "A", "add35", "Сколько будет 3+5?", "8", ["8"]),
    item("add_35_b1", "B", "add35", "Чему равно три плюс пять?", "8", ["8"]),
    item("add_78_a", "A", "add78", "Чему равно 7+8?", "15", ["15"]),
    item("add_78_b1", "B", "add78", "Сколько будет семь плюс восемь?", "15", ["15"]),
    item("mul_96_a", "A", "mul96", "Чему равно 9×6?", "54", ["54"]),
    item("mul_96_b1", "B", "mul96", "Сколько будет 9 умножить на 6?", "54", ["54"]),
    item("c_add_23", "C", "near", "Сколько будет 2+3?", "5", ["5"]),
    item("c_add_45", "C", "near", "Сколько будет 4+5?", "9", ["9"]),
    item("c_add_99", "C", "near", "Сколько будет 9+9?", "18", ["18"]),
    item("c_2plus2_5", "C", "trap", "Два плюс два равно пяти?", "Нет. Два плюс два равно четырём.", ["нет"]),
    # --- calendar / units ---
    item("week_a", "A", "week", "Сколько дней в неделе?", "Семь.", ["семь", "7"]),
    item("week_b1", "B", "week", "Из скольких дней состоит неделя?", "Семь.", ["семь", "7"]),
    item("week_b2", "B", "week", "Сколько дней в одной неделе?", "Семь.", ["семь", "7"]),
    item("year_a", "A", "year", "Сколько месяцев в году?", "Двенадцать.", ["двенадцать", "12"]),
    item("year_b1", "B", "year", "Из скольких месяцев состоит год?", "Двенадцать.", ["двенадцать", "12"]),
    item("day_a", "A", "day", "Сколько часов в сутках?", "Двадцать четыре.", ["двадцать четыре", "24"]),
    item("day_b1", "B", "day", "Сколько часов в одном дне?", "Двадцать четыре.", ["двадцать четыре", "24"]),
    item("c_week8", "C", "trap", "В неделе восемь дней?", "Нет. В неделе семь дней.", ["нет"]),
    item("c_leap", "C", "near", "Сколько дней в високосном году?", "366.", ["366"]),
    item("c_min", "C", "near", "Сколько минут в часе?", "Шестьдесят.", ["шестьдесят", "60"]),
    # --- nature ---
    item("rain_a", "A", "rain", "Что такое дождь?", "Осадки в виде капель воды из облаков.", ["вод", "осад", "капл"]),
    item("rain_b1", "B", "rain", "Объясни, что такое дождь", "Осадки в виде капель воды из облаков.", ["вод", "осад", "капл"]),
    item("rain_b2", "B", "rain", "Дождь — это что?", "Осадки в виде капель воды из облаков.", ["вод", "осад", "капл"]),
    item("snow_a", "A", "snow", "Какого цвета снег?", "Белого.", ["бел"]),
    item("snow_b1", "B", "snow", "Какого цвета снег зимой?", "Белого.", ["бел"]),
    item("snow_b2", "B", "snow", "Снег какого цвета?", "Белого.", ["бел"]),
    item("sky_a", "A", "sky", "Почему небо голубое?", "Воздух рассеивает синий свет сильнее красного.", ["рассеи", "син"]),
    item("sky_b1", "B", "sky", "Из-за чего небо кажется голубым?", "Воздух рассеивает синий свет сильнее красного.", ["рассеи", "син"]),
    item("ice_a", "A", "ice", "Лёд — это что?", "Замёрзшая вода.", ["вод", "замёрз", "замерз"]),
    item("ice_b1", "B", "ice", "Из чего состоит лёд?", "Замёрзшая вода.", ["вод"]),
    item("c_coal", "C", "near", "Какого цвета уголь?", "Чёрного.", ["чёрн", "черн"]),
    item("c_peng", "C", "near", "Пингвины умеют летать?", "Нет.", ["нет"]),
    item("c_flat", "C", "trap", "Земля плоская?", "Нет. Земля шарообразная.", ["нет"]),
    item("c_sun", "C", "near", "Солнце — это звезда?", "Да.", ["да"]),
    item("c_moon", "C", "trap", "Луна — это звезда?", "Нет. Луна — спутник Земли.", ["нет"]),
    # --- identity / chat ---
    item("hi_a", "A", "hi", "Привет", "Привет.", ["привет"]),
    item("hi_b1", "B", "hi", "Здравствуй", "Здравствуй.", ["здравств", "привет"]),
    item("who_a", "A", "who", "Кто ты?", "Я языковая модель JGPT.", ["jgpt"]),
    item("who_b1", "B", "who", "Что ты такое?", "Я языковая модель JGPT.", ["jgpt"]),
    item("name_a", "A", "name", "Как тебя зовут?", "JGPT.", ["jgpt"]),
    item("name_b1", "B", "name", "Твоё имя?", "JGPT.", ["jgpt"]),
    item("c_human", "C", "trap", "Ты человек?", "Нет. Я языковая модель JGPT.", ["нет"]),
    # --- misc facts ---
    item("kg_a", "A", "kg", "Что тяжелее: килограмм пуха или килограмм железа?", "Одинаково: оба по килограмму.", ["одинак", "оба", "килограмм"]),
    item("kg_b1", "B", "kg", "Что тяжелее: килограмм железа или килограмм ваты?", "Одинаково.", ["одинак", "оба"]),
    item("h2o_a", "A", "h2o", "Из чего состоит вода?", "Из водорода и кислорода: H2O.", ["водород", "кислород", "h2o", "h₂o"]),
    item("h2o_b1", "B", "h2o", "Какая химическая формула у воды?", "H2O.", ["h2o", "h₂o"]),
    item("pct_a", "A", "pct", "10% от 200 — это сколько?", "20.", ["20"]),
    item("pct_b1", "B", "pct", "Сколько будет десять процентов от двухсот?", "20.", ["20"]),
    item("rect_a", "A", "rect", "Как найти площадь прямоугольника?", "Умножь длину на ширину.", ["длин", "ширин"]),
    item("rect_b1", "B", "rect", "Площадь прямоугольника как считают?", "Длина умножить на ширину.", ["длин", "ширин"]),
    item("c_spider", "C", "near", "Сколько ног у паука?", "Восемь.", ["восемь", "8"]),
    item("c_insect", "C", "near", "Сколько ног у насекомого?", "Шесть.", ["шесть", "6"]),
    item("hello_a", "A", "hello", "Переведи на русский: hello", "Привет.", ["привет"]),
    item("hello_b1", "B", "hello", "Как по-русски hello?", "Привет.", ["привет"]),
    item("c_thanks", "C", "near", "Переведи на русский: thank you", "Спасибо.", ["спасибо"]),
]


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def hit(answer: str, needles: list[str]) -> bool:
    a = norm(answer)
    for n in needles:
        nn = norm(n)
        if not nn:
            continue
        if nn.isdigit() or nn in {"да", "нет"}:
            if re.search(r"(?<![0-9а-яa-z])" + re.escape(nn) + r"(?![0-9а-яa-z])", a):
                return True
        elif nn in a:
            return True
    return False


def write_gold() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    with GOLD.open("w", encoding="utf-8") as f:
        for row in ITEMS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    PROMPTS.write_text("\n".join(r["user"] for r in ITEMS) + "\n", encoding="utf-8")
    n_a = sum(1 for r in ITEMS if r["cat"] == "A")
    n_b = sum(1 for r in ITEMS if r["cat"] == "B")
    n_c = sum(1 for r in ITEMS if r["cat"] == "C")
    print(f"items={len(ITEMS)} A={n_a} B={n_b} C={n_c} -> {GOLD}")
    print(f"prompts -> {PROMPTS}")


def parse_infer_log(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    blocks = text.split("----- PROMPT -----")
    for block in blocks[1:]:
        if "----- ANSWER -----" not in block:
            continue
        p, rest = block.split("----- ANSWER -----", 1)
        ans = rest.split("----- END -----", 1)[0]
        pairs.append((p.strip(), ans.strip()))
    return pairs


def score(pred_path: Path, out_path: Path | None) -> int:
    gold = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    raw = pred_path.read_text(encoding="utf-8")
    pairs = parse_infer_log(raw)
    if len(pairs) != len(gold):
        print(f"WARN: parsed {len(pairs)} answers, gold {len(gold)}", file=sys.stderr)
    rows = []
    stats = {k: [0, 0] for k in ("A", "B", "C", "ALL")}
    n = min(len(pairs), len(gold))
    for g, (user, ans) in zip(gold[:n], pairs[:n]):
        if user != g["user"]:
            print(f"WARN id={g['id']}: prompt mismatch\n  gold={g['user']!r}\n  got={user!r}", file=sys.stderr)
        ok = hit(ans, g["needles"])
        stats[g["cat"]][0] += int(ok)
        stats[g["cat"]][1] += 1
        stats["ALL"][0] += int(ok)
        stats["ALL"][1] += 1
        rows.append({**g, "pred": ans, "ok": ok})
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"scored -> {out_path}")
    for k in ("A", "B", "C", "ALL"):
        ok, tot = stats[k]
        pct = 100.0 * ok / tot if tot else 0.0
        print(f"{k}: {ok}/{tot} = {pct:.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="write", choices=["write", "score"])
    ap.add_argument("--log", type=Path, help="InferChat stdout for score")
    ap.add_argument("--out", type=Path, help="scored jsonl")
    args = ap.parse_args()
    if args.cmd == "write":
        write_gold()
        return 0
    if not args.log:
        print("score needs --log", file=sys.stderr)
        return 2
    out = args.out or EVAL / "bench_v1_scored.jsonl"
    return score(args.log, out)


if __name__ == "__main__":
    raise SystemExit(main())
