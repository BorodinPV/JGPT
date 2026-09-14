# JGPT Training Runbook

---

## TL;DR — какой скрипт запускать

Скрипты лежат в `scripts/linux/` и `scripts/windows/` ([карта](../scripts/README.md)). Поток 37L SFT: [data-flow-37L-sft.puml](data-flow-37L-sft.puml).

**28L-wide (основной путь, ~134M):** pretrain на `.txt` → SFT → чат. Геометрия `d_model=512`, 32 головы (`d_head=16`), 28 слоёв, seq 1024, BPE 16k без lowercasing и с `<user>`/`<assistant>`. Полный CE по словарю, документы упакованы через `<eos>`, val по документам, dropout 0.1. Чекпоинты **не** пересекаются с 37L и с 20L.

```powershell
.\scripts\windows\jgpt-train-28L-wide.cmd --no-build        # ~1.3 ч/эпоха, 10 эпох; checkpoints\wide_28L_16k_1024
.\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build    # ~2.5 ч на 2 эпохи; checkpoints\wide_28L_sft
.\scripts\windows\jgpt-chat-28L-wide.cmd                    # SFT model_best; --raw --model <path> для претрейна
.\scripts\windows\jgpt-stop-train.cmd                       # остановка (НЕ Ctrl+C)
.\scripts\windows\jgpt-gui.cmd                              # GUI: всё выше + графики, лог, чекпоинты
```

Ориентиры честного val (hold-out по документам, RTX 3080): pretrain 3.16 после 1-й эпохи → 1.47 (ppl 4.4) к 9-й, дальше плато; SFT стартует с ~2.4 и опускается к ~2.0. Сырой претрейн в чате пишет вики-стиль без фактов — это ожидаемо, за формат ответов отвечает SFT.

Стартовый корпус: `python scripts/fetch-ru-pretrain.py` → `data/books/pretrain_txt` (дамп ruwiki). Классика: `--source books`. Полный lib.ru: `scripts/linux/download-lib-ru-library.sh`. SFT-данные: `.jsonl` в `data/sft/raw` → `scripts/sft-filter-short.py` → `data/sft/short` (скрипт SFT делает это сам, если каталог пуст).

**37L SFT ~100M** (JSONL, `env/37L-sft-100M.env`, чекпоинты `checkpoints/sft_37L_16k_2048/`):

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```

```bash
./scripts/linux/jgpt-train-37L-sft.sh
```

Лог: `training_sft_37L.log`. Resume: тот же скрипт без `--fresh`.

**Короткий SFT finetune** (фильтр `data/sft/short`, `env/37L-sft-short-ft.env`, чекпоинты `checkpoints/sft_37L_short_ft/` — исходный 37L не трогает; веса с `model_best.bin`, свежий Adam, LR=1e-4):

```powershell
.\scripts\windows\jgpt-train-37L-sft-short.ps1 --no-build
```

**Экзамен-SFT** (столицы / 2+2 / да-нет, `data/sft/exam`, `env/37L-sft-exam.env`, чекпоинты `checkpoints/sft_37L_exam/`, старт с `sft_37L_short_ft/model_best.bin`):

```powershell
.\scripts\windows\jgpt-train-37L-sft-exam.cmd --no-build
```

**Книги + авто-пресеты** (только Linux, ~35M canonical, `data/books/`):

```bash
./scripts/linux/jgpt-smart.sh
```

Лог: `training_allbooks.log`. Это не единственный launcher — smart только для книг и OOM-монитора.

---

## Как работает авто-адаптация (`jgpt-smart.sh`, Linux)

`scripts/linux/jgpt-smart.sh` — launcher для корпуса книг. Он:
1. Собирает нативную библиотеку (`cmake` + `cmake --build`)
2. Выставляет базовые `JGPT_*` и подмешивает пресет из `env/<имя>.env`
3. Запускает **`AllBooksTrain`** через Maven, лог в `training_allbooks.log`
4. **Bash-монитор** читает лог: OOM, залипание FP16 scale, нет шагов &gt;300 с, плато eval (15 подряд без улучшения) — останавливает JVM и перезапускает со следующим пресетом; upgrade при стабильных улучшениях eval (пороги в начале `jgpt-smart.sh`)

### Иерархия пресетов

| Пресет | Идея | Когда |
|--------|------|-------|
| `00-max-throughput` | batch=2, максимальный throughput | Первая попытка |
| `01-aggressive` | batch=1, агрессивный FP16 | Старт по умолчанию |
| `02-stable` | Мягче FP16, меньше кандидатов CE | При overflow-проблемах |
| `03-recovery` | batch=1, осторожный | При OOM / плато |
| `04-minimal` | последний запасной вариант | После исчерпания 03 |

Направление понижения: `00 → 01 → … → 04`. Повышение — в сторону 00 при стабильных улучшениях eval.

---

## Запуск

### Авто-адаптивный (рекомендуется)
```bash
# Стандарт — с текущего/сохранённого пресета, авто-resume
./scripts/linux/jgpt-smart.sh

# Начать с конкретного пресета
./scripts/linux/jgpt-smart.sh 01-aggressive
./scripts/linux/jgpt-smart.sh 02-stable
```

### Ручной пресет / finetune
```bash
# Явный пресет (тот же smart-скрипт, без смены argv — возьмёт state/current_preset_idx)
./scripts/linux/jgpt-smart.sh 02-stable

# Новый цикл эпох (веса и Adam из чекпоинта, globalStep сбрасывается)
JGPT_FINETUNE=1 ./scripts/linux/jgpt-smart.sh

```

---

## Остановка и продолжение

**Остановить (Windows):**
```powershell
.\scripts\windows\jgpt-stop-train.cmd     # создаёт state\STOP
# в окне обучения дождаться: [STOP] … затем [SHUTDOWN] checkpoint сохранён
```
Или кнопка «Стоп (мягко)» в GUI. **Ctrl+C в PowerShell убивает java без чекпоинта** — потеряете шаги после последнего `checkpoint_step_N` (каждые `JGPT_SAVE_EVERY_STEPS`, в wide-пресетах 200).

**Остановить (Linux):** Ctrl+C / SIGTERM — shutdown hook в `LLMTrainer` сохраняет `checkpoint_final.bin`.

**Продолжить:** тот же скрипт без флагов. Тренер выбирает чекпоинт с наибольшим `globalStep` среди `final / step_N / epoch_N / best`, поэтому resume работает и после жёсткого обрыва.

```powershell
.\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build
```

**Смена плана** (другие данные, другой пресет, другой loss) — один раз `--restart-plan`: веса + Adam остаются, шаг/LR/эпоха/best сбрасываются. Не путать с `--fresh` (архив каталога и с нуля).

---

## Добавление книг в процессе

1. Положить `.txt` в `data/books/pretrain_txt` (28L-wide) или `data/books/` (smart)
2. Мягко остановить (см. выше)
3. Запустить снова:

```powershell
# Продолжить с того же шага (LR-расписание не сбрасывается; новые документы попадут в поток при следующей эпохе):
.\scripts\windows\jgpt-train-28L-wide.cmd --no-build

# Новый цикл эпох с расширенным корпусом (веса и Adam остаются):
.\scripts\windows\jgpt-train-28L-wide.cmd --no-build --restart-plan
```

```bash
# Linux, книги + smart
./scripts/linux/jgpt-smart.sh
JGPT_FINETUNE=1 ./scripts/linux/jgpt-smart.sh
```

> **Если добавлено много новых книг** с незнакомой лексикой — пересоздать токенизатор:
> ```bash
> rm checkpoints/tokenizer_global.bin
> ./scripts/linux/jgpt-smart.sh  # пересоздаст словарь (~2 мин)
> ```

---

## Мониторинг

```powershell
# GUI (Windows): вкладки Обучение / Лог / Чекпоинты / Чат, обновление раз в 2 с из state\stats.json
.\scripts\windows\jgpt-gui.cmd
```

```bash
# Хвост лога без PERF/VRAM-шума
tail -f training_28L_wide.log | grep -E "\[STEP\]|\[EVAL\]|\[CKPT\]|\[FP16\]|WARN|SMART"

# Текущий шаг и пресет (smart)
cat state/last_step.txt
cat state/current_preset_idx

# Старый HTML-дашборд: тот же stats.json, но нужен http-сервер
python -m http.server 8765   # → http://localhost:8765/docs/dashboard.html
```

Здоровый прогон: `val_loss` падает на каждом eval; `[FP16] scale` колеблется 32768↔65536; в `stats.json` `skipped_steps`, `non_finite`, `oom_errors`, `fp16_stuck` равны 0; train с dropout выше val на 0.1–0.2 — норма, не переобучение.

---

## Расшифровка проблем в логе

| Строка в логе | Причина | Авто-ответ `jgpt-smart.sh` |
|---------------|---------|----------------------------------|
| `cudaMalloc failed` / `OutOfMemoryError` | VRAM переполнен | Downgrade → следующий пресет |
| `overflow-скип` много раз | FP16 scale залип | Downgrade после 8 уникальных шагов |
| Нет `[STEP]` больше 300 с | Зависание | Downgrade |
| Плато eval 15 раз подряд | Нет прогресса | Downgrade |
| 30 улучшений eval подряд | Стабильный прогресс | Upgrade → быстрый пресет |
| `[SMART] Фатальная CUDA-ошибка` | GPU-контекст повреждён | `exit(2)`, перезапустить JVM |

---

## Производительность (RTX 3080)

| Режим | Throughput | Шаг |
|-------|------------|-----|
| 28L-wide pretrain, seq 1024, batch 4×16, full CE + dropout | ~12 000 tok/s | ~5.3 с на 65 536 токенов |
| 28L-wide SFT, batch 4×8 | ~12 000 tok/s | ~2.7 с |
| 37L SFT, seq 2048, sampled CE | ~9 000–10 000 tok/s | ~27–28 с на 262 144 токена |
| 12L, пресет `02-stable`, книги | ~26 000 tok/s | ~1250 мс |

28L-wide на sampled CE давал ~6 000 tok/s: сборка кандидатов на CPU и gather-голова без Tensor Cores дороже одного полного GEMM `4096×512×16000`. Разбивка шага пишется в лог при `JGPT_TRAIN_PERF=1` (`[PERF] … прямой / лосс+∂CE / обратн / клип+опт`).

### Ключевые оптимизации

- **FlashAttention-2** — cuDNN SDPA, fused FP16; fallback WMMA
- **Full-vocab CE на GPU** — логиты и ∂CE на устройстве, один GEMM на голову
- **Warp-level reduction** — для embedding gradients
- **cuBLAS GEMM** — FP16 Tensor Cores для всех матричных операций
- **CUDA Graph** — на уровне декодер-слоёв (в wide-пресетах выключен; при dropout отключается сам)

## Ключевые параметры (в `env/*.env`)

| Переменная | Описание |
|------------|----------|
| `JGPT_TRAIN_LOSS_MODE` | `full` (по умолчанию для wide) или `sampled` (старые пресеты; `JGPT_SAMPLED_CE_CANDIDATES` читается только в этом режиме) |
| `JGPT_BATCH_SIZE`, `JGPT_ACCUMULATION_STEPS` | Микробатч × накопление = эффективный батч |
| `JGPT_DROPOUT` | Residual + embedding dropout на GPU-пути (0 = выкл., по умолчанию выкл.) |
| `JGPT_SAVE_EVERY_STEPS`, `JGPT_EVAL_EVERY_STEPS` | Частота `checkpoint_step_N` и eval (wide: 200 / 100) |
| `JGPT_INTERACTIVE_EVERY` | Генерация сэмпла каждые N шагов. **Держите 0**: каждый сэмпл делит FP16 loss scale на 64, и он уходит в 1 |
| `JGPT_FP16_DYNAMIC_INITIAL`, `JGPT_FP16_DYNAMIC_MAX`, `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | Динамический loss scale |
| `JGPT_VAL_FRACTION` | Доля hold-out (по документам / диалогам) |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | CUDA graph на декодер-слой (1/0); не совместим с dropout |
| `JGPT_FINETUNE` | `1` = сброс шага/LR/эпохи/best при загрузке чекпоинта (то же, что `--restart-plan`) |

---

## Dropout и weight decay

Включаются пресетом (`JGPT_DROPOUT=0.1` в `28L-wide-*.env`); без переменной dropout выключен.

| Где | Реализация |
|-----|------------|
| residual после attention `W_o` и после FFN `W_2` | GPU-ядро, inverted dropout, маска пересчитывается в backward по тому же seed (шаг + слой + место) — ничего не хранится |
| embedding (token + pos) | то же |
| attention weights | на GPU не реализован (0) |

Eval и генерация — всегда без dropout. AdamW weight decay действует только на матрицы (ранг ≥ 2); RMSNorm-гейны не затухают.

### Рекомендации

- Менять `JGPT_DROPOUT` только вместе с `--fresh` или `--restart-plan`
- Переобучение (train ↓, val ↑) — поднять до 0.2
- Недообучение — 0.05 или 0

---

## Структура файлов

```
JGPT/
├── env/
│   ├── 28L-wide-pretrain.env   ← основной pretrain (full CE, dropout, packing)
│   ├── 28L-wide-sft.env        ← SFT после него
│   ├── 20L-wide-*.env, 37L-*.env
│   └── 00…04-*.env             ← цепочка jgpt-smart.sh
├── state/
│   ├── stats.json              ← метрики текущего прогона (GUI / dashboard.html)
│   ├── STOP                    ← запрос мягкой остановки (появляется на время остановки)
│   ├── last_step.txt           ← последний сохранённый шаг
│   └── current.env, current_preset_idx ← только smart
├── scripts/
│   ├── windows/                ← jgpt-train-*.cmd/.ps1, jgpt-chat-*.cmd, jgpt-stop-train.cmd, jgpt-gui.cmd, build-cuda.ps1
│   ├── linux/                  ← bash: smart, 28L/20L/37L, build-cuda.sh
│   ├── sft-filter-short.py, fetch-ru-pretrain.py, …
│   └── README.md
├── src/main/java/com/veles/llm/jgpt/gui/ ← JavaFX GUI (собирается jgpt-gui.cmd в target/gui-classes)
├── data/books/pretrain_txt/    ← корпус 28L-wide (ruwiki + классика)
├── data/sft/raw/ → data/sft/short/ ← .jsonl для SFT
├── checkpoints/tokenizer_wide_16k.bin
├── checkpoints/wide_28L_16k_1024/ ← 28L-wide pretrain (checkpoint_*/model_*/tokenizer_*)
├── checkpoints/wide_28L_sft/   ← 28L-wide SFT
├── checkpoints/*_prev_backup/  ← архив после --fresh
├── docs/dashboard.html         ← старый веб-дашборд (state/stats.json)
├── training_28L_wide.log, training_28L_wide_sft.log
└── training_allbooks.log, training_sft_37L.log ← старые пути
```
