# JGPT Training Runbook

---

## TL;DR — Одна команда для запуска всего

```bash
./scripts/jgpt-smart.sh
```

Всё остальное делается автоматически.

---

## Как работает авто-адаптация

`jgpt-smart.sh` — единственный launcher в `scripts/`. Он:
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
./scripts/jgpt-smart.sh

# Начать с конкретного пресета
./scripts/jgpt-smart.sh 01-aggressive
./scripts/jgpt-smart.sh 02-stable
```

### Ручной пресет / finetune
```bash
# Явный пресет (тот же smart-скрипт, без смены argv — возьмёт state/current_preset_idx)
./scripts/jgpt-smart.sh 02-stable

# Новый цикл эпох (веса и Adam из чекпоинта, globalStep сбрасывается)
JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh

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
./scripts/jgpt-smart.sh   # подхватит checkpoint_final.bin автоматически
```

---

## Добавление книг в процессе

1. Положить `.txt` в `data/books/`
2. `Ctrl+C`
3. Запустить снова:

```bash
# Продолжить с того же шага (LR-расписание не сбрасывается):
./scripts/jgpt-smart.sh

# Начать новый цикл эпох с расширенным корпусом:
JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh
```

> **Если добавлено много новых книг** с незнакомой лексикой — пересоздать токенизатор:
> ```bash
> rm checkpoints/tokenizer_global.bin
> ./scripts/jgpt-smart.sh  # пересоздаст словарь (~2 мин)
> ```

---

## Мониторинг

```bash
# Хвост лога (основной «дашборд»)
tail -f training_allbooks.log

# Веб-дашборд с графиками (открыть в браузере)
xdg-open dashboard.html
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

## Производительность (RTX 3080, пресет 02-stable)

| Метрика | Значение |
|---------|----------|
| Throughput | ~26 000 tokens/sec |
| Время шага | ~1250 мс (forward 600 + CE 9 + backward 620 + optimizer 35) |
| VRAM | ~5.2 GB / 10 GB |

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
├── data/books/                 ← .txt файлы для обучения
├── checkpoints/all_books/      ← веса, чекпоинты, токенизатор
├── dashboard.html              ← веб-дашборд (открыть в браузере)
└── training_allbooks.log       ← лог (append, не перезаписывается)
```
