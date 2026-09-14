# Запуск обучения и пресеты (AllBooksTrain)

Таблица переменных окружения: **[README.md](../../README.md)** в корне репозитория.

## Запуск

| Способ | Команда |
|--------|---------|
| **28L-wide pretrain ~134M** (Windows) | `.\scripts\windows\jgpt-train-28L-wide.cmd --no-build` |
| **28L-wide SFT** (Windows) | `.\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build` |
| То же (Linux) | `./scripts/linux/jgpt-train-28L-wide.sh`, `./scripts/linux/jgpt-train-28L-wide-sft.sh` |
| **GUI** (Windows) | `.\scripts\windows\jgpt-gui.cmd` — старт/стоп, графики, лог, чекпоинты, чат |
| 37L SFT ~100M (старый путь) | `.\scripts\windows\jgpt-train-37L-sft.ps1` / `./scripts/linux/jgpt-train-37L-sft.sh` |
| Книги + авто-адаптация (Linux) | `./scripts/linux/jgpt-smart.sh` |
| Напрямую Maven | `mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain -Dexec.args='--boo . --data-dir <dir>'` (после CUDA-сборки и `JGPT_*`) |

Карта скриптов: [scripts/README.md](../../scripts/README.md).

**Производительность (RTX 3080 10 GB):** 28L-wide full CE ~12k tok/s (шаг 5.3 с на 65 536 токенов); 37L SFT seq 2048 ~9–10k; 12L `02-stable` ~26k.

### Геометрия модели

| Параметр | 28L-wide (`env/28L-wide-*.env`) | canonical (`LLMConfig.canonical()`) |
|----------|---------------------------------|-------------------------------------|
| vocab | 16000 (`checkpoints/tokenizer_wide_16k.bin`, без lowercasing, `<user>`/`<assistant>`) | 8000 |
| seq | 1024 | 1024 |
| d_model | 512 | 384 |
| heads | 32 (d_head = 16) | 24 (d_head = 16) |
| слои | 28 | 12 |
| SwiGLU d_intermediate | 2048 | 1536 |
| параметры | ~134M | ~34.9M |

`JGPT_MAX_SEQ_LEN` / `JGPT_PRESET_NUM_LAYERS` / `JGPT_D_MODEL` / `JGPT_NUM_HEADS` в `env/*.env` должны совпадать с чекпоинтом — иначе `shape mismatch` при загрузке весов. GUI берёт геометрию для чата из пресета, чей `JGPT_CHECKPOINT_SUBDIR` совпадает с каталогом модели.

### Loss и данные (28L-wide)

- `JGPT_TRAIN_LOSS_MODE=full` — CE по всему словарю. Sampled CE (384 равномерных негативов из 16k) почти никогда не штрафует правдоподобные, но неверные токены: модель выучивает стиль, а не факты; к тому же gather-голова медленнее одного GEMM.
- Pretrain: документы кодируются целиком и пакуются в один поток через `<eos>` (короткие статьи и хвосты не теряются). Train/val делится **по документам** (`JGPT_VAL_FRACTION=0.05`), а не по окнам — иначе val утекает из тех же книг.
- SFT: `JGPT_SFT=1`, loss только на ответах ассистента, один диалог на окно (`JGPT_SFT_PACK=one`), split по уникальному вопросу (`JGPT_SFT_SPLIT=dialog`).

### Как работает `jgpt-smart.sh`

Один скрипт в `scripts/linux/`:
1. Собирает `libjgpt_cuda_extra.so` и `libjgpt_cuda.so` (`cmake` + `cmake --build`)
2. Выставляет базовые `JGPT_*` env-переменные и подмешивает активный пресет из `env/<имя>.env`
3. Запускает **`AllBooksTrain`** с `tee -a training_allbooks.log`
4. **Bash-монитор** по логу: OOM, залипание FP16 scale, «зависание» без шагов, плато eval — при необходимости останавливает JVM и перезапускает со следующим пресетом (upgrade/downgrade по порогам в начале скрипта)

Цепочка имён пресетов и переключение — только в bash: массив `PRESETS` и логика монитора в `jgpt-smart.sh`.

### Finetune (сброс `globalStep`, веса и Adam из чекпоинта)

```bash
JGPT_FINETUNE=1 ./scripts/linux/jgpt-smart.sh
# или явный пресет:
JGPT_FINETUNE=1 ./scripts/linux/jgpt-smart.sh 01-aggressive
```

## Цепочка пресетов

`00-max-throughput` → `01-aggressive` → `02-stable` → `03-recovery` → `04-minimal`

| Пресет | Идея |
|--------|------|
| **00** | batch=2, максимальный throughput; первый кандидат на OOM |
| **01** | Агрессивный режим, старт по умолчанию |
| **02** | Мягче FP16, стабильнее при overflow |
| **03** | batch=1, осторожный режим |
| **04** | минимальный — последний запасной вариант |

## Пороги bash-монитора (`jgpt-smart.sh`)

**Downgrade** — при первом же из:
- OOM / фатальная CUDA-ошибка (любой момент)
- ≥ **8** уникальных шагов оптимизатора с overflow-скипами
- **15** eval подряд без улучшения best loss (плато)
- Нет шага оптимизатора **300 с** (зависание)

**Upgrade** — **30** улучшений eval подряд при текущем индексе > 0

**Стартовый пресет**: из `state/current_preset_idx`, иначе **01-aggressive**.

## Пресет и env

Файлы `env/<имя>.env` содержат только `export JGPT_*=…`. Скрипт записывает активный пресет в `state/current_preset_idx` и обновляет symlink `state/current.env → ../env/<имя>.env`.

`JGPT_*` экспортируются в окружение JVM через `jgpt-smart.sh` ДО запуска Maven — подпроцессы наследуют их.

### Канонический GPU-train

При доступной CUDA обучение всегда идёт полным VRAM-путём (resident + decoder pipeline + device CE/backward). Отдельные `JGPT_TRAIN_GPU_RESIDENT` / `JGPT_FULL_GPU_TRAIN` / `JGPT_GPU_E2E_TRAIN` / `JGPT_DEVICE_LOGITS_TRAIN` / `JGPT_DEVICE_DECODER_BWD` / `JGPT_DECODER_GPU_PIPELINE` больше не выбирают путь. Периодический VRAM cleanup/trim выключен (`JGPT_VRAM_CLEANUP_EVERY_STEPS=0`, `JGPT_CUDA_TRIM_EVERY_STEPS=0`); барьеры после eval/sample остаются.

## Resume, чекпоинты и остановка

- Каталог — `JGPT_CHECKPOINT_SUBDIR` пресета (`checkpoints/wide_28L_16k_1024`, `checkpoints/wide_28L_sft`, у книг — `checkpoints/all_books`).
- Файлы: `checkpoint_<tag>.bin` (веса + Adam + позиция в эпохе + FP16 loss-scale, формат **v5**) и парные `model_<tag>.bin` / `tokenizer_<tag>.bin`. Теги: `final` (мягкая остановка / конец плана), `step_N` (каждые `JGPT_SAVE_EVERY_STEPS`, хранятся два последних), `epoch_N`, `best` (лучший val).
- Запись атомарная (`.tmp` → rename): жёсткий обрыв не оставляет битого файла.
- **Resume** — тот же скрипт без флагов: берётся чекпоинт с наибольшим `globalStep` среди всех тегов, веса — из парного `model_*.bin`. При обрыве теряется не больше `JGPT_SAVE_EVERY_STEPS` шагов.
- **Остановка на Windows** — только `scripts\windows\jgpt-stop-train.cmd` (файл `state/STOP`; тренер дописывает `checkpoint_final` и выходит) или кнопка «Стоп» в GUI. Ctrl+C в окне PowerShell убивает java без чекпоинта. На Linux Ctrl+C/SIGTERM работают через shutdown hook.
- `--restart-plan` (= `JGPT_FINETUNE=1` на один запуск): веса и Adam из чекпоинта, но `globalStep`, LR-расписание (warmup + cosine), индекс эпохи и best сбрасываются. Нужен один раз после смены objective/данных/пресета; дальше — обычный resume. Если `JGPT_FINETUNE` остался в shell, скрипт предупредит.
- `--fresh`: содержимое каталога переносится в `*_prev_backup`, токенизатор не трогается.
- Веса содержат таблицу позиционных эмбеддингов → `JGPT_MAX_SEQ_LEN` должен совпадать с чекпоинтом.

## Dropout и weight decay

Включаются пресетом: `JGPT_DROPOUT=0.1` (28L-wide pretrain и SFT). По умолчанию (без переменной) dropout **выключен** — старые пресеты и тесты детерминированы.

| Где | Значение | Реализация |
|-----|----------|------------|
| residual после attention `W_o` и после FFN `W_2` | `JGPT_DROPOUT` | GPU-ядро, inverted dropout (`× 1/(1-p)`), маска восстанавливается в backward по тому же seed — ничего не хранится |
| embedding (token + pos) | `JGPT_DROPOUT` | то же |
| attention weights | 0 | на GPU-пути не реализован |

Seed — функция шага, слоя и места (attn/FFN/embed), поэтому forward и backward видят одну маску. Пока dropout активен, CUDA Graph на слои декодера отключается (маска меняется каждый шаг). Eval и генерация всегда без dropout.

AdamW weight decay применяется только к тензорам ранга ≥ 2 (матрицы); gain'ы RMSNorm и прочие 1-D параметры не затухают.

**Рекомендации**: менять `JGPT_DROPOUT` — только вместе с `--fresh` или `--restart-plan`; при переобучении (train ↓, val ↑) поднять до 0.2; если недообучение — 0.05 или 0.

## Книги и токенизатор

- Тексты: **`data/books/**/*.txt`**
- Добавили книги → Ctrl+C → положили файлы → снова `./scripts/linux/jgpt-smart.sh`
- Токенизатор: **`checkpoints/tokenizer_global.bin`**. Удалить для пересоздания при следующем старте

## Состояние и мониторинг

| Файл | Содержимое |
|------|-----------|
| `state/stats.json` | Метрики текущего прогона (пишет `TrainingStatsWriter` на каждом шаге): ряды train/val loss, perplexity, ток/с, счётчики overflow/OOM |
| `state/STOP` | Запрос мягкой остановки (создаёт `jgpt-stop-train.cmd` или GUI) |
| `state/last_step.txt` | Последний сохранённый globalStep |
| `state/current_preset_idx`, `state/current.env` | Только для `jgpt-smart.sh` |
| `training_28L_wide.log`, `training_28L_wide_sft.log`, `training_allbooks.log`, … | Полный лог прогона (append), имя — в шапке соответствующего `.ps1`/`.sh` |

```powershell
# GUI: вкладки Обучение (графики, ETA, старт/стоп), Лог (фильтры STEP/EVAL/CKPT/FP16/WARN), Чекпоинты, Чат
.\scripts\windows\jgpt-gui.cmd
```

```bash
# Хвост лога без PERF/VRAM-шума
tail -f training_28L_wide.log | grep -E "\[STEP\]|\[EVAL\]|\[CKPT\]|\[FP16\]|WARN"

# Старый HTML-дашборд (тот же stats.json; нужен http-сервер, file:// блокирует fetch)
python -m http.server 8765   # → http://localhost:8765/docs/dashboard.html
```

Что смотреть в логе: `[EVAL] … val_loss=` на честном hold-out; `[FP16] scale` должен ходить между 32768 и 65536 (`÷2 после eval` — норма, `÷64 после генерации` — значит включён `JGPT_INTERACTIVE_EVERY`, scale уйдёт в 1 — выключите его); `overflow`/`OOM` в stats.json должны оставаться нулями.

## Ручной запуск без обёрток

```bash
# Загрузить пресет вручную и запустить тот же цикл, что в smart (cmake + mvn allbooks):
set -a; source env/01-aggressive.env; set +a
./scripts/linux/jgpt-smart.sh 01-aggressive
```

**Важно**: `JGPT_*` должны быть экспортированы в **той же** shell-сессии, что запускает Maven.
