# JGPT

GPT-модель (decoder-only transformer) с **полным обучением на GPU** через JNI + CUDA/cuBLAS.

> **~35M параметров** · **FlashAttention-2** · **FP16 Tensor Cores** · **CUDA Graph**

---

## 🚀 Быстрый старт

```bash
# Сборка + обучение (одна команда)
./scripts/jgpt-smart.sh

# С конкретным пресетом
./scripts/jgpt-smart.sh 02-stable
```

Положите `.txt` файлы в `data/books/` — и обучение начнёт с них.

---

## 📐 Модель

| Параметр | Значение |
|---|---|
| Архитектура | Decoder-only GPT, pre-norm |
| Параметры | ~34.9M |
| d_model | 384 |
| Слои | 12 |
| Attention heads | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 768) |
| Контекст | 1024 токена |
| Токенизация | BPE, vocab = 8000 |
| Позиции | RoPE |
| Нормализация | RMSNorm |

---

## ⚡ GPU-ускорение

- **FP16 GEMM** — cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** — fused QKV attention
- **Fused-операции** — RMSNorm + FFN, RMSNorm + LM head
- **Полный GPU-цикл** — forward, backward, optimiser — всё на VRAM
- **Decoder pipeline** — слой-за-слоем без D2H
- **CUDA Graph** — на слои декодера (опционально)
- **Async checkpointing** — веса пишутся в фоне

---

## 🎛 Пресеты обучения

| Пресет | Batch | FP16 scale | Кандидаты | Когда |
|---|---|---|---|---|
| `00-max-throughput` | 2 | 65536 | 1024 | Макс. скорость |
| `01-aggressive` | 1 | 65536 | 1024 | Старт по умолчанию |
| **`02-stable`** | **1** | **32768** | **512** | **При overflow** |
| `03-recovery` | 1 | 16384 | 256 | OOM / плато |
| `04-minimal` | 1 | 8192 | 128 | Запасной |

> `accumulation_steps = 8` для всех пресетов. Эффективный batch = 8192 токена.

### Авто-адаптация

`jgpt-smart.sh` мониторит лог и автоматически:
- **Downgrade** при OOM, FP16 overflow, плато eval
- **Upgrade** при 30 последовательных улучшениях eval
- **Sticky mode** — не переключает пресет (`JGPT_SMART_STICKY_PRESET=1`)

---

## 📊 Производительность

**RTX 3080 (10 GB VRAM)**, пресет 02-stable:

| Метрика | Значение |
|---|---|
| Tokens/sec | ~11 500 |
| Шаг | ~715 мс (forward 368 + CE 4 + backward 314 + optimiser 29) |
| VRAM | ~3700 / 9873 МБ |

---

## 📁 Структура

```
├── src/main/java/     # Java: модель, трейнер, данные
├── src/main/cpp/      # CUDA/C++: ядра, JNI, optimiser
├── env/               # Пресеты (00–04)
├── scripts/           # Запускные скрипты
├── data/books/        # .txt для обучения
├── checkpoints/       # model_*.bin, checkpoint_*.bin
├── docs/dashboard.html  # Веб-дашборд (открыть в браузере)
├── state/             # stats.json, current_preset_idx
└── logs/              # training_allbooks.log
```

---

## 💾 Чекпоинты

### checkpoint_*.bin — состояние обучения
```
veles.ckpt.v4
├─ globalStep, bestLoss, epoch, seqIndex
└─ Adam m/v buffers
```

### model_*.bin — веса модели
```
veles.weights.v1
├─ num_tensors (int)
└─ для каждого: shape_len, shape[], data (float[], BigEndian)
```

### tokenizer_global.bin
BPE-словарь (Java serialization).

---

## 🔄 Resume / Finetune

| Сценарий | Команда |
|---|---|
| Продолжить обучение | `./scripts/jgpt-smart.sh` |
| Новые эпохи (веса + Adam сохраняются) | `JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh` |
| Добавить книги | `.txt` → `data/books/` → перезапуск |
| Пересоздать токенизатор | `rm checkpoints/tokenizer_global.bin` → перезапуск |

---

## 📈 Мониторинг

```bash
tail -f logs/training_allbooks.log | grep -E "\[STEP\]|\[EVAL\]|WARN|SMART"
xdg-open docs/dashboard.html    # графики в браузере, автообновление 5с
cat state/last_step.txt         # последний шаг
cat state/stats.json            # полная статистика
```

### Loss

- **sampled_train_loss** — CE на 512 кандидатов (не full-vocab, всегда ниже)
- **eval_loss** — full-vocab CE на hold-out validation (5% окон)
- Метрики **несравнимы напрямую**

---

## 🔧 Переменные окружения

Основные переменные (остальные — в `env/02-stable.env`):

### Обучение
| Переменная | Описание |
|---|---|
| `JGPT_BATCH_SIZE` | Микробатч (1–2) |
| `JGPT_ACCUMULATION_STEPS` | Шагов накопления (8) |
| `JGPT_MAX_SEQ_LEN` | Длина контекста (1024) |
| `JGPT_EPOCHS` | Число эпох |
| `JGPT_LEARNING_RATE` | Базовый LR (косинус с разогревом 10%) |
| `JGPT_TRAIN_LOSS_MODE` | `full` или `sampled` |
| `JGPT_SAMPLED_CE_CANDIDATES` | Кандидатов на строку (512) |

### GPU
| Переменная | Описание |
|---|---|
| `JGPT_FULL_GPU_TRAIN` | Полный GPU-цикл (1) |
| `JGPT_FP16_MATMUL` | FP16 + dynamic loss scale (1) |
| `JGPT_FLASH_ATTENTION` | FlashAttention-2 (1) |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | CUDA Graph на декодер (0) |
| `JGPT_DEVICE_LOGITS_TRAIN` | CE + LM head backward на device (1) |
| `JGPT_DEVICE_DECODER_BWD` | Decoder backward на VRAM (1) |

### Кэши и буферы
| Переменная | Описание |
|---|---|
| `JGPT_BATCH_PINNED` | Pinned host-память (1) |
| `JGPT_BATCH_PREFETCH` | Фоновый prefetch батча (1) |
| `JGPT_BLOCK_CACHE_POOL` | Пул кэша блоков ⚠️ (1) |
| `JGPT_BLOCK_CACHE_GROW_ONLY` | Кэш только растёт ⚠️ (1) |
| `JGPT_ACTIVATION_CACHE_FP16` | FP16 кэш активаций (1) |

> ⚠️ **Важно:** `BLOCK_CACHE_POOL` и `BLOCK_CACHE_GROW_ONLY` **не совместимы**. При `GROW_ONLY=1` пул отключается автоматически (см. [фикс](https://github.com/...)).

### FP16 Loss Scale
| Переменная | Значение |
|---|---|
| `JGPT_FP16_DYNAMIC_INITIAL` | 32768 |
| `JGPT_FP16_DYNAMIC_MAX` | 65536 |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | 100 шагов |

### Отладка
| Переменная | Описание |
|---|---|
| `JGPT_DEBUG_GPU_TRAIN` | JSONL debug-режим (0) |
| `JGPT_TRAIN_PERF` | Расширенные `[PERF]` логи (1) |
| `JGPT_TIMINGS` | Краткие тайминги (0) |
| `JGPT_PROFILE` | TrainingProfiler (0) |

### JVM properties
Альтернатива env-переменным через `-Djgpt.*`:
```bash
-Djgpt.fullGpuTrain=true -Djgpt.allow.no.gpu=true -Djgpt.fused.ffn.rms.w1w3=true
```

Полный справочник — в JavaDoc `LLMConfig` и `env/02-stable.env`.

---

## 🛠 Сборка

```bash
# Java (Maven)
mvn compile

# CUDA (CMake)
mkdir -p build && cd build
cmake ../src/main/cpp
make -j$(nproc)
```

Требования: Java 17+, CUDA 12.x, cuBLAS, GCC ≤ 13 (или `-allow-unsupported-compiler`).
