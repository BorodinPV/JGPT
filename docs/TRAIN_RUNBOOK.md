# JGPT Training Runbook

---

## TL;DR — какой скрипт запускать

Скрипты лежат в `scripts/linux/` и `scripts/windows/` ([карта](../scripts/README.md)).

**37L SFT ~100M** (JSONL, `env/37L-sft-100M.env`, чекпоинты `checkpoints/sft_37L_16k_2048/`):

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```

```bash
./scripts/linux/jgpt-train-37L-sft.sh
```

Лог: `training_sft_37L.log`. Resume: тот же скрипт без `--fresh`.

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

**Остановить:**
```bash
Ctrl+C
# Shutdown hook в LLMTrainer сохраняет checkpoint_final.bin
```

**Продолжить:**
```bash
./scripts/linux/jgpt-smart.sh   # подхватит checkpoint_final.bin автоматически
```

---

## Добавление книг в процессе

1. Положить `.txt` в `data/books/`
2. `Ctrl+C`
3. Запустить снова:

```bash
# Продолжить с того же шага (LR-расписание не сбрасывается):
./scripts/linux/jgpt-smart.sh

# Начать новый цикл эпох с расширенным корпусом:
JGPT_FINETUNE=1 ./scripts/linux/jgpt-smart.sh
```

> **Если добавлено много новых книг** с незнакомой лексикой — пересоздать токенизатор:
> ```bash
> rm checkpoints/tokenizer_global.bin
> ./scripts/linux/jgpt-smart.sh  # пересоздаст словарь (~2 мин)
> ```

---

## Мониторинг

```bash
# Хвост лога (основной «дашборд»)
tail -f training_allbooks.log

# Веб-дашборд с графиками (открыть в браузере)
xdg-open docs/dashboard.html
# Автообновление каждые 30 с из state/stats.json

# Хвост лога
tail -f training_allbooks.log | grep -E "\[STEP\]|\[EVAL\]|\[SAMPLE\]|WARN|SMART"

# Текущий шаг и пресет
cat state/last_step.txt
cat state/current_preset_idx
```

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

| Режим | Throughput |
|-------|------------|
| 12L, пресет `02-stable`, книги | ~26 000 tokens/sec |
| 37L SFT, seq 2048 | ~9 000–10 000 tokens/sec |

12L шаг ~1250 мс, VRAM ~5.2 / 10 GB. 37L шаг ~27–28 с на эффективный батч 262144 токена.

### Ключевые оптимизации

- **FlashAttention-2** — tile size 128, полностью fused attention
- **Optimized CE** — block-per-row kernel, 12x faster (~110ms → ~9ms)
- **Warp-level reduction** — для embedding gradients, 32x less atomic contention
- **CUDA Graph** — на уровне декодер-слоёв, уменьшает CPU launch overhead
- **cuBLAS GEMM** — FP16 Tensor Cores для всех матричных операций

## Ключевые параметры (в `env/*.env`)

| Переменная | Описание |
|------------|----------|
| `JGPT_BATCH_SIZE` | Размер батча |
| `JGPT_SAMPLED_CE_CANDIDATES` | Кандидаты sampled CE |
| `JGPT_FP16_DYNAMIC_INITIAL` | Начальный loss scale |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | Интервал роста scale |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | CUDA graph на декодер-слой (1/0). Включено по умолчанию — даёт +5-10% скорости |

---

## Dropout регуляризация

Dropout включён по умолчанию для предотвращения переобучения. Работает автоматически во время обучения, отключается при инференсе.

| Тип | Значение по умолчанию | Куда применяется |
|-----|----------------------|------------------|
| `residualDropout` | 0.1 (10%) | После FFN перед residual connection |
| `attentionDropout` | 0.1 (10%) | После attention output перед residual connection |
| `embeddingDropout` | 0.1 (10%) | На embedding слое (резерв) |

### Как это работает

- **Inverted dropout**: случайно обнуляет 10% элементов, остальные масштабируются на `1/(1-p) = 1.11`, чтобы сумма сохранялась
- **XOR-shift RNG**: быстрый генератор случайных чисел в CUDA ядре с seed-based воспроизводимостью
- **По слоям**: каждый decoder block использует свой seed (`42 + layerIdx * 1000`)
- **Без переменных окружения**: dropout настраивается через `TrainingConfig`, работает автоматически

### Рекомендации

- **Не включайте dropout на середине обучения** — начните заново с чистого чекпоинта
- **При переобучении** (train loss ↓, eval loss ↑) можно увеличить dropout до 0.2-0.3
- **Если модель недообучается** — уменьшите dropout до 0.05 или отключите (0.0)

---

## Структура файлов

```
JGPT/
├── env/
│   ├── 00-max-throughput.env
│   ├── 01-aggressive.env       ← старт по умолчанию
│   ├── 02-stable.env
│   └── 03-recovery.env
├── state/
│   ├── current.env             ← symlink на активный пресет
│   ├── current_preset_idx      ← индекс пресета (0–3)
│   ├── last_step.txt           ← последний сохранённый шаг
│   └── stats.json              ← метрики для dashboard.html
├── scripts/
│   ├── linux/                  ← bash: smart, 24/32/37L, build-cuda.sh
│   ├── windows/                ← ps1: 37L-sft, build-cuda.ps1
│   └── README.md
├── data/books/                 ← .txt для jgpt-smart / 24L / 32L
├── data/sft/raw/               ← .jsonl для 37L-sft
├── checkpoints/all_books/      ← книги
├── checkpoints/sft_37L_16k_2048/
├── docs/dashboard.html         ← веб-дашборд (state/stats.json)
├── training_allbooks.log       ← smart
└── training_sft_37L.log        ← 37L SFT
```
