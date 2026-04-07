# Запуск обучения и пресеты (AllBooksTrain)

Таблица переменных окружения: **[README.md](../../README.md)** в корне репозитория.

## Запуск

| Способ | Команда |
|--------|---------|
| **Рекомендуемый** (авто-адаптация, один JVM) | `./scripts/jgpt-smart.sh` |
| С явного пресета | `./scripts/jgpt-smart.sh 01-aggressive` |
| Напрямую через Maven | `mvn -q compile exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.AllBooksTrain -Dexec.args='--boo .'` (после `cmake` и экспорта `JGPT_*`) |

### Как работает `jgpt-smart.sh`

Один скрипт в `scripts/`:
1. Собирает `libjgpt_cuda.so` (`cmake` + `cmake --build`)
2. Выставляет базовые `JGPT_*` env-переменные и подмешивает активный пресет из `env/<имя>.env`
3. Запускает **`AllBooksTrain`** с `tee -a training_allbooks.log`
4. **Bash-монитор** по логу: OOM, залипание FP16 scale, «зависание» без шагов, плато eval — при необходимости останавливает JVM и перезапускает со следующим пресетом (upgrade/downgrade по порогам в начале скрипта)

Цепочка имён пресетов и переключение — только в bash: массив `PRESETS` и логика монитора в `jgpt-smart.sh`.

### Finetune (сброс `globalStep`, веса и Adam из чекпоинта)

```bash
JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh
# или явный пресет:
JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh 01-aggressive
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

## Resume, чекпоинты и `JGPT_MAX_SEQ_LEN`

- Чекпоинты: **`checkpoints/all_books/`** (`checkpoint_final.bin` приоритетнее `checkpoint_epoch_N.bin`)
- Checkpoint сохраняется через **shutdown hook** в `LLMTrainer` (Ctrl+C, SIGTERM, supervisedStop)
- Веса содержат размер позиционных эмбеддингов → **`JGPT_MAX_SEQ_LEN` должен совпадать** с тем, на котором сохранялся чекпоинт
- `JGPT_FINETUNE=1`: сбрасывается только `globalStep`; веса и Adam остаются. Задайте вместе с `./scripts/jgpt-smart.sh` (см. выше).

## Книги и токенизатор

- Тексты: **`data/books/**/*.txt`**
- Добавили книги → Ctrl+C → положили файлы → снова `./scripts/jgpt-smart.sh`
- Токенизатор: **`checkpoints/tokenizer_global.bin`**. Удалить для пересоздания при следующем старте

## Состояние и мониторинг

| Файл | Содержимое |
|------|-----------|
| `state/last_step.txt` | Последний сохранённый globalStep |
| `state/current_preset_idx` | Текущий индекс пресета (0–4) |
| `state/current.env` | Symlink на активный env-файл |
| `state/stats.json` | Метрики для веб-дашборда (пишет `TrainingStatsWriter`) |
| `training_allbooks.log` | Полный лог (append) |

```bash
# Лог обучения
tail -f training_allbooks.log

# Веб-дашборд с графиками (Chart.js, автообновление 30 с)
xdg-open dashboard.html
```

## Ручной запуск без обёрток

```bash
# Загрузить пресет вручную и запустить тот же цикл, что в smart (cmake + mvn allbooks):
set -a; source env/01-aggressive.env; set +a
./scripts/jgpt-smart.sh 01-aggressive
```

**Важно**: `JGPT_*` должны быть экспортированы в **той же** shell-сессии, что запускает Maven.
