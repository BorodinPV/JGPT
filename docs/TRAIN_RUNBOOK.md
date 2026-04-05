# JGPT Training Runbook

---

## TL;DR — Одна команда для запуска всего

```bash
./scripts/jgpt-smart.sh
```

Всё остальное делается автоматически.

---

## Как работает авто-адаптация

```
jgpt-smart.sh
    │
    ├── запускает обучение (train-e2e-gpu.sh allbooks) в фоне
    ├── каждые 30 секунд проверяет лог на проблемы
    │
    ├── OOM обнаружен?           → TERM → resume с 03-recovery
    ├── FP16 stuck=1.0× ≥8 раз? → TERM → resume с 02-stable
    ├── Зависание 5 минут?       → TERM → resume с 02-stable
    │
    └── Обучение завершилось штатно → конец
```

При каждом переключении обучение **останавливается через shutdown hook** (checkpoint сохраняется) и **автоматически возобновляется** с новым пресетом.

### Иерархия пресетов

| Пресет | Пакет/токены/с | Когда |
|--------|---------------|-------|
| `00-max-throughput` | batch=2, ~14k т/с | Первая попытка — максимум |
| `01-aggressive` | batch=1, ~7k т/с | Стабильная работа (по умолчанию) |
| `02-stable` | candidates=256, ~5k т/с | При FP16 overflow |
| `03-recovery` | seq=512, ~4k т/с | При OOM |

Направление: `00 → 01 → 02 → 03` (авто при проблемах).

---

## Запуск

### Авто-адаптивный (рекомендуется)
```bash
# Стандарт — с текущего пресета, авто-resume
./scripts/jgpt-smart.sh

# Попробовать максимальную скорость (batch=2, может OOM → авто-fallback)
./scripts/jgpt-smart.sh 00-max-throughput

# Начать с надёжного пресета
./scripts/jgpt-smart.sh 01-aggressive

# Новый цикл эпох (веса сохраняются, LR-расписание сбрасывается)
./scripts/jgpt-smart.sh --finetune
```

### Ручной
```bash
# Простой запуск с текущим пресетом
./scripts/jgpt-start.sh

# Явный пресет
./scripts/jgpt-start.sh 02-stable

# С мониторингом в отдельном терминале
./scripts/jgpt-monitor.sh &
./scripts/jgpt-start.sh
```

---

## Остановка и продолжение

**Остановить:**
```bash
Ctrl+C
# jgpt-smart.sh: сам остановит Java и сохранит checkpoint
# jgpt-start.sh: shutdown hook в Java сохранит checkpoint при SIGTERM/SIGINT
```

**Продолжить:**
```bash
./scripts/jgpt-smart.sh   # авто-resume, подхватит checkpoint_final.bin
```

---

## Добавление книг в процессе

1. Положить `.txt` в `data/books/`
2. Остановить: `Ctrl+C`
3. Запустить снова:

```bash
# Продолжить с того же шага (LR-расписание не сбрасывается):
./scripts/jgpt-smart.sh

# Начать новый цикл эпох с расширенным корпусом:
./scripts/jgpt-smart.sh --finetune
```

> **Если добавлено много новых книг** с незнакомой лексикой — пересоздать токенизатор:
> ```bash
> rm checkpoints/tokenizer_global.bin
> ./scripts/jgpt-smart.sh  # пересоздаст словарь (~2 мин)
> ```
> Размер vocab (8000) сохраняется, модель совместима.

---

## Мониторинг

```bash
# Живой дашборд (отдельный терминал)
./scripts/jgpt-monitor.sh

# Следить за прогрессом в логе
tail -f training_allbooks.log | grep -E "\[STEP\]|\[EVAL\]|\[SAMPLE\]|WARN|SMART"

# Последний шаг
cat state/last_step.txt

# Текущий пресет
readlink state/current.env
```

---

## Ручное переключение пресета

```bash
# Переключить пресет (следующий перезапуск применит его)
ln -sf ../env/02-stable.env state/current.env

# Или через jgpt-start.sh
./scripts/jgpt-start.sh 02-stable
```

---

## Расшифровка проблем в логе

| Строка в логе | Причина | Авто-ответ smart |
|---------------|---------|-----------------|
| `cudaMalloc failed` | VRAM переполнен | Понижение до `03-recovery` |
| `масштаб loss ... 1.000×` (много раз) | FP16 scale залип | Понижение до `02-stable` |
| `Ранний останов: patience` | 20+ eval без улучшения | Штатное завершение, resume |
| `Non-finite gradient` раз в 100 шагов | Норма после eval | Ничего |
| `[FP16] scale ... → ...` (колеблется) | Норма с growth=50 | Ничего |

---

## Ключевые параметры

| Переменная | Пресет 01 | Описание |
|------------|-----------|----------|
| `JGPT_MAX_SEQ_LEN` | 1024 | Контекст. Снизить при OOM |
| `JGPT_SAMPLED_CE_CANDIDATES` | 512 | Кандидаты sampled CE |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | 50 | **Должно быть < 100** (eval каждые 100 шагов) |
| `JGPT_FP16_AUX_SOFTEN` | 0 | Выключить доп. деление scale |
| `JGPT_EARLY_STOP_EVAL_PATIENCE` | 20 | Eval без улучшения до стопа |
| `JGPT_INTERACTIVE_EVERY` | 500 | Генерация каждые N шагов |
| `JGPT_SAMPLE_PROMPT` | (встроенные) | Свои промпты через `\|` |

---

## Структура файлов

```
JGPT/
├── env/
│   ├── 00-max-throughput.env   ← batch=2 (экспериментальный)
│   ├── 01-aggressive.env       ← рабочий (~7k т/с)
│   ├── 02-stable.env           ← при FP16 проблемах
│   └── 03-recovery.env         ← при OOM
├── state/
│   ├── current.env             ← symlink на активный пресет
│   ├── current_preset_idx      ← индекс пресета (для smart)
│   ├── last_step.txt           ← последний шаг (auto)
│   └── .need_downgrade         ← флаг от монитора
├── data/books/                 ← .txt книги для обучения
├── checkpoints/all_books/      ← веса, чекпоинты, токенизатор
└── training_allbooks.log       ← лог (append, не перезаписывается)
```
