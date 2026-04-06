# Запуск обучения и пресеты (AllBooksTrain)

Полный справочник переменных окружения `JGPT_*`, `-Djgpt.*` и смежных флагов: **[README.md](../../README.md)** в корне репозитория.

## Запуск

| Способ | Команда |
|--------|---------|
| **Рекомендуемый** (авто-адаптация, один JVM) | `./scripts/jgpt-smart.sh` |
| С явного пресета | `./scripts/jgpt-smart.sh 01-aggressive` |
| Ручной (без авто-переключения) | `./scripts/jgpt-start.sh [пресет] [--finetune]` |
| Напрямую через Maven | `mvn -q exec:java -Dexec.mainClass=com.veles.llm.jgpt.app.SmartTrainingSupervisor -Dexec.args="--boo . [пресет]"` |

### Как работает `jgpt-smart.sh`

Тонкая bash-обёртка (~60 строк):
1. Собирает `libjgpt_cuda.so` (`cmake` + `cmake --build`)
2. Выставляет базовые `JGPT_*` env-переменные
3. Запускает **один JVM-процесс**: `SmartTrainingSupervisor | tee -a training_allbooks.log`

Вся логика адаптации — в Java:

- **`SmartTrainingSupervisor`** запускает `AllBooksTrain.runWithPreset(preset)` в треде и слушает события
- **`TrainingEventCallback`** — интерфейс событий из `LLMTrainer`: каждый шаг оптимизатора, каждый eval, overflow-скипы, OOM
- **`PresetDecider`** накапливает метрики и выдаёт решение `DOWNGRADE / UPGRADE / NONE`
- При решении: `LLMTrainer.requestSupervisedStop()` → ждём завершения → меняем пресет → снова `runWithPreset()` (тот же JVM, тот же чекпоинт)
- При фатальной CUDA-ошибке: `System.exit(2)` — GPU-контекст повреждён, JVM нельзя переиспользовать

Конфигурация цепочки: [`PresetConfig.SMART_PRESET_CHAIN`](../../src/main/java/com/veles/llm/jgpt/training/PresetConfig.java)

### `jgpt-start.sh` — ручной запуск

Загружает `env/<пресет>.env` в текущий shell → запускает `train-e2e-gpu.sh allbooks`. Без авто-переключения пресетов. Полезен для:
- Явного выбора пресета без smart-логики
- `--finetune` (сброс `globalStep` при сохранённых весах и Adam-состоянии)

## Цепочка пресетов

`00-max-throughput` → `01-aggressive` → `02-stable` → `03-recovery`

| Пресет | Идея |
|--------|------|
| **00** | batch=2, максимальный throughput; первый кандидат на OOM |
| **01** | Агрессивный режим, старт по умолчанию |
| **02** | Мягче FP16, стабильнее при overflow |
| **03** | batch=1, самый осторожный — аварийное продолжение |

## Пороги PresetDecider

**Downgrade** — при первом же из:
- OOM / фатальная CUDA-ошибка (любой момент)
- ≥ **8** уникальных шагов оптимизатора с overflow-скипами
- **15** eval подряд без улучшения best loss (плато)
- Нет шага оптимизатора **300 с** (зависание)

**Upgrade** — **30** улучшений eval подряд при текущем индексе > 0

**Стартовый пресет**: из `state/current_preset_idx`, иначе **01-aggressive**.

## Пресет и env

Файлы `env/<имя>.env` содержат только `export JGPT_*=…`. `SmartTrainingSupervisor` записывает активный пресет в `state/current_preset_idx` и обновляет symlink `state/current.env → ../env/<имя>.env`.

`JGPT_*` экспортируются в окружение JVM через `jgpt-smart.sh` ДО запуска Maven — подпроцессы наследуют их.

## Resume, чекпоинты и `JGPT_MAX_SEQ_LEN`

- Чекпоинты: **`checkpoints/all_books/`** (`checkpoint_final.bin` приоритетнее `checkpoint_epoch_N.bin`)
- Checkpoint сохраняется через **shutdown hook** в `LLMTrainer` (Ctrl+C, SIGTERM, supervisedStop)
- Веса содержат размер позиционных эмбеддингов → **`JGPT_MAX_SEQ_LEN` должен совпадать** с тем, на котором сохранялся чекпоинт
- `--finetune` / `JGPT_FINETUNE=1`: сбрасывается только `globalStep`; веса и Adam остаются. Работает через `jgpt-start.sh`, не через `jgpt-smart.sh`

## Книги и токенизатор

- Тексты: **`data/books/**/*.txt`**
- Добавили книги → Ctrl+C → положили файлы → снова `./scripts/jgpt-smart.sh`
- Токенизатор: **`checkpoints/tokenizer_global.bin`**. Удалить для пересоздания при следующем старте

## Состояние и мониторинг

| Файл | Содержимое |
|------|-----------|
| `state/last_step.txt` | Последний сохранённый globalStep |
| `state/current_preset_idx` | Текущий индекс пресета (0–3) |
| `state/current.env` | Symlink на активный env-файл |
| `state/stats.json` | Метрики для веб-дашборда (пишет `TrainingStatsWriter`) |
| `training_allbooks.log` | Полный лог (append) |

```bash
# Живой терминальный дашборд
./scripts/jgpt-monitor.sh

# Веб-дашборд с графиками (Chart.js, автообновление 30 с)
xdg-open dashboard.html
```

## Ручной запуск без обёрток

```bash
# Загрузить пресет вручную и запустить:
set -a; source env/01-aggressive.env; set +a
./scripts/train-e2e-gpu.sh allbooks
```

**Важно**: `JGPT_*` должны быть экспортированы в **той же** shell-сессии, что запускает Maven.
