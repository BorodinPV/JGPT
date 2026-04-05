#!/usr/bin/env python3
"""Парсит training_allbooks.log и пишет state/stats.json для HTML-дашборда."""
import json, re, os, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
LOG  = ROOT / "training_allbooks.log"
STATE = ROOT / "state"
OUT  = STATE / "stats.json"

if not LOG.exists():
    print(f"Лог не найден: {LOG}", file=sys.stderr)
    sys.exit(1)

lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()

# ── Временны́е ряды ───────────────────────────────────────────
eval_steps, eval_loss, perplexity = [], [], []
train_steps, train_loss = [], []
overflow_steps = []

last_step = 0

for line in lines:
    # [STEP] — запоминаем текущий шаг и train loss
    m = re.search(r'\[STEP\].*?шаг (\d+).*?sampled_train_loss=([0-9,]+)', line)
    if m:
        step = int(m.group(1))
        loss = float(m.group(2).replace(',', '.'))
        last_step = step
        train_steps.append(step)
        train_loss.append(loss)
        continue

    # [EVAL] перплексия + loss идут подряд — привязываем оба к last_step
    m = re.search(r'\[EVAL\] перплексия: ([0-9,]+)', line)
    if m:
        perplexity.append(float(m.group(1).replace(',', '.')))
        continue

    m = re.search(r'\[EVAL\] эпоха.*?loss=([0-9.]+)', line)
    if m:
        eval_steps.append(last_step)
        eval_loss.append(float(m.group(1)))
        continue

    # Пропущенные шаги
    m = re.search(r'Шаг (\d+) пропущен: переполнение', line)
    if m:
        overflow_steps.append(int(m.group(1)))
        continue

# ── Скалярные метрики ─────────────────────────────────────────
def last_match(pattern, text, group=1, default=""):
    matches = re.findall(pattern, text)
    return matches[-1] if matches else default

log_text = "\n".join(lines)

current_step  = int(last_match(r'шаг (\d+)', log_text) or 0)
total_steps   = int(last_match(r'прогресс шагов \d+/(\d+)', log_text) or 0)
current_epoch = last_match(r'эпоха (\d+/\d+)', log_text, default="1/20")
best_loss     = float(last_match(r'лучший сохранённый=([0-9.]+)', log_text) or 0)
# Обрезаем перплексию до длины eval_steps (первая запись может быть без шага)
min_len = min(len(eval_steps), len(perplexity))
eval_steps = eval_steps[:min_len]
eval_loss  = eval_loss[:min_len]
perplexity = perplexity[:min_len]

last_eval     = eval_loss[-1] if eval_loss else 0
last_perp     = perplexity[-1] if perplexity else 0
last_train    = train_loss[-1] if train_loss else 0
tokens_s      = int(last_match(r'ток/с≈(\d+)', log_text) or 0)
lr_raw        = last_match(r'lr=([0-9,]+e[+-]\d+)', log_text, default="")
lr            = lr_raw.replace(',', '.') if lr_raw else "-"
skipped       = len(overflow_steps)
non_finite    = log_text.count("Non-finite gradient")
oom           = sum(1 for p in ["cudaMalloc failed","OutOfMemoryError","out of memory"] if p in log_text)
fp16_stuck    = len(re.findall(r'масштаб loss.*?1\.000×', log_text))

# ── Генерации ─────────────────────────────────────────────────
samples_raw = re.findall(r'\[SAMPLE\] сгенерировано: (.+)', log_text)
sample_steps_raw = re.findall(r'\[SAMPLE\] промежуточная генерация:.*?шаг (\d+)', log_text)
samples = [{"text": t.strip()} for t in samples_raw[-5:]]
last_sample = samples_raw[-1].strip() if samples_raw else ""
last_sample_step = sample_steps_raw[-1] if sample_steps_raw else ""

# ── Конфиг из current.env ─────────────────────────────────────
env_file = STATE / "current.env"
preset = "?"
preset_idx = "?"
cfg = {"seq_len": "-", "batch": "1", "fp16_max": "-",
       "fp16_grow_interval": "-", "early_stop_patience": "-",
       "ce_candidates": "-", "description": ""}

if env_file.exists():
    env_target = env_file.resolve()
    preset = env_target.stem
    idx_file = STATE / "current_preset_idx"
    preset_idx = idx_file.read_text().strip() if idx_file.exists() else "?"

    env_text = env_file.read_text(encoding="utf-8", errors="replace")

    def env_val(key, default="-"):
        m = re.search(rf'^export {key}=(\S+)', env_text, re.MULTILINE)
        if m:
            val = m.group(1).split('#')[0].strip()
            return val if val else default
        return default

    cfg["seq_len"]              = env_val("JGPT_MAX_SEQ_LEN")
    cfg["batch"]                = env_val("JGPT_BATCH_SIZE", "1")
    cfg["fp16_max"]             = env_val("JGPT_FP16_DYNAMIC_MAX")
    cfg["fp16_grow_interval"]   = env_val("JGPT_FP16_DYNAMIC_GROWTH_INTERVAL")
    cfg["early_stop_patience"]  = env_val("JGPT_EARLY_STOP_EVAL_PATIENCE")
    cfg["ce_candidates"]        = env_val("JGPT_SAMPLED_CE_CANDIDATES")
    desc_lines = [l.lstrip('# ') for l in env_text.splitlines()
                  if l.startswith('#') and ('Пресет' in l or '~' in l)]
    cfg["description"] = "  ".join(desc_lines[:2])

# ── Финальный JSON ────────────────────────────────────────────
data = {
    "updated":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "current_step":   current_step,
    "total_steps":    total_steps,
    "current_epoch":  current_epoch,
    "best_loss":      best_loss,
    "last_eval_loss": last_eval,
    "last_perplexity":last_perp,
    "last_train_loss":last_train,
    "tokens_per_sec": tokens_s,
    "lr":             lr,
    "skipped_steps":  skipped,
    "non_finite":     non_finite,
    "oom_errors":     oom,
    "fp16_stuck":     fp16_stuck,
    "preset":         preset,
    "preset_idx":     preset_idx,
    "config":         cfg,
    "last_sample":    last_sample,
    "last_sample_step": last_sample_step,
    "samples":        samples,
    "eval_steps":     eval_steps,
    "eval_loss":      eval_loss,
    "perplexity":     perplexity,
    "train_steps":    train_steps,
    "train_loss":     train_loss,
    "overflow_steps": overflow_steps,
}

OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"stats.json обновлён: {OUT}  ({data['updated']})")
print(f"  eval: {len(eval_steps)} точек  train: {len(train_steps)} точек  overflow: {skipped}")
