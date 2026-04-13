# JGPT — GPT-модель с GPU-обучением

Transformer-модель архитектуры GPT (decoder-only) с полным обучением на GPU через JNI + CUDA/cuBLAS.

---

## Быстрый старт

```bash
# Одна команда: сборка + обучение на всех книгах из data/books/
./scripts/jgpt-smart.sh

# С конкретным пресетом
./scripts/jgpt-smart.sh 02-stable
```

Подробности запуска — [TRAIN_RUNBOOK.md](docs/TRAIN_RUNBOOK.md).

---

## Архитектура модели

| Параметр | Значение |
|----------|----------|
| Тип | Decoder-only GPT |
| Эмбеддинг токенов | BPE, vocab = 8000 |
| Context length | 1024 |
| d_model | 384 |
| Attention heads | 24 (d_head = 16) |
| Decoder layers | 12 |
| FFN intermediate | SwiGLU (d_intermediate = 2 × d_model = 768) |
| Позиционное кодирование | RoPE |
| Нормализация | RMSNorm (pre-norm) |
| Параметры | ~34.9M |

---

## Stack

| Компонент | Технология |
|-----------|------------|
| JVM | Java 17+, Maven |
| CUDA | CUDA 12.x, cuBLAS, FP16 Tensor Cores |
| Native | C++/CUDA JNI (`src/main/cpp/`) |
| Оптимизатор | AdamW, fused GPU-шаг |
| Mixed precision | Динамический FP16 loss scale (32768× → 65536×) |

### GPU-ускорение

- **FP16 GEMM** через cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** (требует d_head = 16)
- **Fused-операции**: RMSNorm + FFN (W1+W3 strided-batched), RMSNorm + LM head
- **Полный GPU-цикл**: веса, прямой/обратный проход, оптимизатор — всё на VRAM
- **Decoder GPU pipeline**: слой-за-слоем без D2H между слоями
- **CUDA Graph** на слои декодера (опционально, env `JGPT_DECODER_LAYER_CUDA_GRAPH=1`)
- **Async checkpointing**: веса пишутся в фоновом потоке

---

## Пресеты обучения

| Пресет | Batch | Accum | FP16 st | Кандидаты | Назначение |
|--------|-------|-------|---------|-----------|------------|
| `00-max-throughput` | 2 | 8 | 65536 | 1024 | Максимальный throughput |
| `01-aggressive` | 1 | 8 | 65536 | 1024 | Старт по умолчанию |
| **`02-stable`** | **1** | **8** | **32768** | **512** | **При overflow** |
| `03-recovery` | 1 | 8 | 16384 | 256 | OOM / плато |
| `04-minimal` | 1 | 8 | 8192 | 128 | Запасной |

Текущий активный: **02-stable** (`state/current_preset_idx`).

### Авто-адаптация (`jgpt-smart.sh`)

Скрипт мониторит лог и автоматически:
- **Downgrade** при OOM, залипании FP16, зависании, плато eval
- **Upgrade** при 30 последовательных улучшениях eval
- **Sticky mode** (`JGPT_SMART_STICKY_PRESET=1`) — не переключает пресет

---

## Производительность

На RTX 3080 (9873 МБ VRAM), пресет 02-stable:

| Метрика | Значение |
|---------|----------|
| Tokens/sec | ~11 500 |
| Время шага | ~715 мс |
| ├ Forward | 368 мс |
| ├ CE loss | 4 мс |
| ├ Backward | 314 мс |
| └ Clip + Adam | 29 мс |
| VRAM занято | ~3700 / 9873 МБ |
| Эффективный batch | 8192 токенов (1 × 8 × 1024) |

---

## Структура проекта

```
JGPT/
├── src/
│   ├── main/java/.../jgpt/
│   │   ├── app/          # AllBooksTrain, InferChat
│   │   ├── model/        # GPTModel, DecoderBlock, TokenEmbedding
│   │   ├── training/     # LLMTrainer, Checkpoint IO, Config
│   │   ├── data/         # DataLoader, BPETokenizer
│   │   └── TensorOpsGPU  # JNI-обёртки
│   └── main/cpp/
│       ├── jgpt_cuda.cu           # Основные CUDA-ядра и JNI
│       ├── jgpt_cuda_extra.cu     # CE, attention, optimiser JNI
│       └── *.cuh, *.inl, *.h      # Заголовки и inline
├── env/                    # Пресеты (02-stable.env, ...)
├── scripts/
│   └── jgpt-smart.sh       # Главный launcher
├── data/books/             # .txt файлы для обучения
├── checkpoints/all_books/  # model_final.bin, checkpoint_final.bin, ...
├── state/                  # stats.json, current_preset_idx, last_step.txt
├── dashboard.html          # Веб-дашборд (открыть в браузере)
├── pom.xml                 # Maven
└── CMakeLists.txt (cpp)    # Сборка нативной библиотеки
```

---

## Формат чекпоинтов

### checkpoint_*.bin (состояние обучения)
```
veles.ckpt.v4
├─ globalStep      (int)
├─ bestLoss        (float)
├─ epoch           (int)
├─ seqIndex        (int)
└─ Adam m/v buffers
```

### model_*.bin (веса модели)
```
veles.weights.v1
├─ num_tensors     (int)
└─ для каждого:
    ├─ shape_len   (int)
    ├─ shape[]     (int × shape_len)
    └─ data        (float[], BigEndian)
```

### tokenizer_global.bin
BPE-словарь (Java serialization, `BPETokenizer.save/load`).

---

## Resume / Finetune

| Сценарий | Команда |
|----------|---------|
| Продолжить с последнего шага | `./scripts/jgpt-smart.sh` |
| Новый цикл эпох (веса + Adam сохраняются) | `JGPT_FINETUNE=1 ./scripts/jgpt-smart.sh` |
| Добавить книги | Положить `.txt` в `data/books/` → перезапуск |
| Пересоздать токенизатор | `rm checkpoints/tokenizer_global.bin` → перезапуск |
| Если globalStep ≥ плана | `JGPT_IF_STEP_BEYOND_PLAN=restart_schedule` |

---

## Мониторинг

```bash
tail -f training_allbooks.log | grep -E "\[STEP\]|\[EVAL\]|WARN|SMART"
xdg-open dashboard.html    # графики в браузере, автообновление 30с
cat state/last_step.txt    # последний шаг
cat state/current_preset_idx  # текущий пресет
```

### Интерпретация loss

- **sampled_train_loss** — CE на подмножестве 512 кандидатов (не full-vocab)
- **eval_loss (val_loss)** — full-vocab CE на hold-out validation (5% окон)
- Эти метрики **несравнимы напрямую** — sampled CE всегда ниже

---

## Переменные окружения

Полный справочник всех `JGPT_*` переменных — см. раздел ниже и `env/02-stable.env`.

Парные system properties `-Djgpt.*` описаны в JavaDoc `LLMConfig`.

---

# Справочник переменных окружения JGPT_*

Список **`JGPT_*`**, которые читает код (через `System.getenv` или обёртки вроде `readPositiveEnvInt("JGPT_…", …)`). Скрипт **`./scripts/jgpt-smart.sh`** подмешивает файлы `env/<пресет>.env` в окружение процесса перед Maven — JVM наследует те же переменные.

Многие флаги имеют **парные system properties `jgpt.*`** (например `JGPT_FULL_GPU_TRAIN` ↔ `-Djgpt.fullGpuTrain`): если в описании не указано иное, см. JavaDoc в `LLMConfig` и соответствующих классах.

Полный перечень литеральных `System.getenv("JGPT_…")` в исходниках **и** тестах поддерживается в `src/test/resources/jgpt-training-env-keys.txt` (тест `JgptTrainingEnvCatalogTest`).

## Основные (обучение)

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_BATCH_SIZE` | — | Размер микробатча. См. `LLMConfig.applyBatchSizeOverrideFromEnv`. |
| `JGPT_ACCUMULATION_STEPS` | — | Микробатчей градиента на шаг оптимизатора (минимум 1). |
| `JGPT_MAX_SEQ_LEN` | — | Макс. длина контекста. Должна согласовываться с чекпоинтом. |
| `JGPT_EPOCHS` | 20 (smart50M) | Число эпох поверх значения в `LLMConfig`. |
| `JGPT_PRESET_NUM_LAYERS` | — | Число декодер-слоёв (целое > 0). |
| `JGPT_LEARNING_RATE` / `JGPT_LR` | из конфига | Базовый learning rate (косинус с разогревом 10%). |
| `JGPT_TRAIN_LOSS_MODE` | `sampled` | `full` или `sampled` (train; eval — всегда full CE). |
| `JGPT_SAMPLED_CE_CANDIDATES` | 512 | Кандидатов на строку в sampled train loss (минимум 2). |
| `JGPT_SAMPLED_CE_NEGATIVE_MODE` | `BATCH_SHARED_UNIFORM` | Режим негативов для sampled CE. |
| `JGPT_VAL_FRACTION` | 0 | Доля окон под hold-out validation (0–0.5). |
| `JGPT_VAL_SEED` | 42 | Seed для split train/validation. |
| `JGPT_TRAIN_SHUFFLE_SEED` | 42 | Seed перемешивания батчей на каждой эпохе. |
| `JGPT_INTERACTIVE_EVERY` | 0 | Каждые N шагов — генерация текста; 0 = выкл. |
| `JGPT_SAMPLE_PROMPT` | авто | Промпт для генерации; несколько вариантов через `\|`. |
| `JGPT_EXIT_AFTER_STEP` | 0 | Завершить JVM после N шагов оптимизатора; 0 = без ограничения. |
| `JGPT_FINETUNE` | 0 | `1`/`true`: загрузить веса и Adam, сбросить `globalStep`. |
| `JGPT_IF_STEP_BEYOND_PLAN` | `skip` | Если globalStep из чекпоинта ≥ нового плана: `skip` / `restart_schedule` / `fail`. |

## GPU / CUDA

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_TRAIN_GPU_RESIDENT` | 1 | Резидентные веса на GPU; `0`/`false` выключает. |
| `JGPT_FULL_GPU_TRAIN` | 1 | Полный шаг обучения на GPU. Свойство: `jgpt.fullGpuTrain`. |
| `JGPT_GPU_E2E_TRAIN` | 1 | End-to-end GPU (resident + full step + device logits/decoder). |
| `JGPT_DECODER_GPU_PIPELINE` | 1 | Декодер на GPU слой-за-слоем (нужен для полного GPU train). |
| `JGPT_DEVICE_LOGITS_TRAIN` | 1 | CE и backward LM head на device. |
| `JGPT_DEVICE_DECODER_BWD` | 1 | Backward декодера на VRAM. |
| `JGPT_FP16_MATMUL` | 1 | Matmul и путь в FP16; включает динамический loss scale. |
| `JGPT_FLASH_ATTENTION` | 1 | FlashAttention-2 в CUDA (требует d_head = 16). |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | 0 | CUDA Graph на полный слой декодера. |
| `JGPT_CUDA_LIB` | — | Абсолютный путь к `libjgpt_cuda.so`. |
| `JGPT_ALLOW_NO_GPU` | — | Разрешить работу без CUDA (тесты). |
| `JGPT_CE_ASYNC` | 0 | Асинхронный CE на GPU; с sampled обычно `0`. |
| `JGPT_CE_GPU_MIN_ELEMENTS` | — | Порог размера; ниже CE может остаться на CPU. |

## FP16 Loss Scale

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_FP16_DYNAMIC_INITIAL` | 32768 | Начальное значение dynamic loss scale. |
| `JGPT_FP16_DYNAMIC_MAX` | 65536 | Верхняя граница loss scale. |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | 100 | Шагов между попытками увеличить loss scale. |
| `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK` | 256 | После стольких overflow на min scale — сброс к начальному. |
| `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH` | 0 | Сбрасывать loss scale в начале каждой эпохи. |
| `JGPT_FP16_AUX_SOFTEN` | 1 | Мастер-флаг: 0 отключает деление loss scale после вспомогательной GPU-работы. |
| `JGPT_FP16_AUX_SOFTEN_EVAL` | 2 | Делитель loss scale после eval. |
| `JGPT_FP16_AUX_SOFTEN_SAMPLE` | — | Делитель loss scale после генерации. |
| `JGPT_AMP_GROWTH_INTERVAL` | — | Приоритетнее `_GROWTH_INTERVAL`. См. `DynamicLossScaler`. |

## Кэши и буферы

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_BATCH_DIRECT` | — | Off-heap прямой буфер для батча. |
| `JGPT_BATCH_PINNED` | 1 | Pinned host-память для батча. |
| `JGPT_BATCH_PREFETCH` | 1 | Фоновая подготовка следующего батча. |
| `JGPT_BLOCK_CACHE_POOL` | 1 | Thread-local пул кэша блоков. |
| `JGPT_BLOCK_CACHE_POOL_MAX` | — | Макс. объектов в очереди пула. |
| `JGPT_BLOCK_CACHE_GROW_ONLY` | 1 | Кэш активаций только растёт, не сжимается. |
| `JGPT_BLOCK_CACHE_MAX_BYTES` | 0 | Мягкий лимит кэша; 0 = без лимита. |
| `JGPT_ACTIVATION_CACHE_FP16` | 1 | Хранить слоты кэша активаций в FP16. |

## Fused-операции

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_FUSED_FFN_RMS_W1W3` | 1 | Один JNI: RMSNorm + проекции SwiGLU W1/W3. |
| `JGPT_FUSED_LM_HEAD` | 1 | Слияние финального RMSNorm и LM head на GPU. |
| `JGPT_GENERATE_GPU_KV` | — | Генерация с KV-кэшем на GPU. |

## Early Stopping

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_EARLY_STOP_EVAL_PATIENCE` | 0 | Eval подряд без улучшения; 0 = выкл. |
| `JGPT_EARLY_STOP_OVERFIT` | 0 | `0`/`false` — отключить останов по train↓ eval↑. |

## Профилирование и отладка

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_PROFILE` | 0 | Включить `TrainingProfiler`. |
| `JGPT_PROFILE_STEPS` | — | Число шагов профилирования. |
| `JGPT_TIMINGS` | 0 | Краткие тайминги в строках train loss. |
| `JGPT_TRAIN_PERF` | 1 | Расширенные строки `[PERF]` (фазы, ток/с). |
| `JGPT_TRAIN_VRAM_STEP_PROBE` | 0 | NDJSON-снимки VRAM вокруг decoder forward. |
| `JGPT_TRAIN_VRAM_STEP_PROBE_EVERY` | 50 | Интервал пробников VRAM. |
| `JGPT_DEBUG_GPU_TRAIN` | 0 | Отладочный режим GPU-обучения (JSONL/события). |
| `JGPT_DEBUG_GPU_TRAIN_LOG` | — | Путь/режим файла лога для debug. |
| `JGPT_DEBUG_CURSOR_B39372` | — | Pre-launch-лог для CUDA graph отладки. |
| `JGPT_DEBUG_NDJSON_LOG` | — | Путь NDJSON-файла для debug-сессии. |
| `JGPT_DECODER_CUDA_GRAPH_MEM_LOG` | — | Лог VRAM после graph launch. |
| `JGPT_DECODER_GRAPH_MIN_FREE_MIB` | 0 | Минимум free VRAM перед graph-path; 0 = выкл. |
| `JGPT_DECODER_LAYER_CUDA_GRAPH_LOG` | 0 | Подробные логи указателей decoder graph. |
| `JGPT_BWD_LAYER_FINITE_CHECK` | 0 | Проверки finite после слоёв backward. |
| `JGPT_RMSNORM_EPS` | 1e-5 / 1e-6 | Epsilon RMSNorm (1e-5 при FP16). |

## Статистика и пресеты

| Переменная | По умолчанию | Описание |
|------------|-------------|----------|
| `JGPT_STATS_JSON` | true | Запись `state/stats.json`; `0` = выкл. |
| `JGPT_STATS_MAX_SERIES` | 20 000 | Макс. точек в eval/train рядах (10–500 000). |
| `JGPT_STATS_PRESET` | — | Подпись пресета в статистике. |
| `JGPT_STATS_PRESET_IDX` | — | Индекс пресета в статистике. |
| `JGPT_SMART_STICKY_PRESET` | 0 | Не переключать пресет автоматически. |
| `JGPT_CKPT_PRUNE` | — | Автоудаление старых чекпоинтов; 0 = выкл. |
| `JGPT_CKPT_KEEP_STEP_SNAPSHOTS` | 2 | Сколько step-снимков хранить. |
| `JGPT_CKPT_KEEP_EPOCH_SNAPSHOTS` | 2 | Сколько epoch-снимков хранить. |

## Тесты и зонды (не влияют на train)

| Переменная | Описание |
|------------|----------|
| `JGPT_BATCH_PROBE` | Только тесты: opt-in в `EffectiveBatchProbeTest`. |
| `JGPT_BATCH_PROBE_CPU` | Тест: форсировать CPU. |
| `JGPT_BATCH_PROBE_FP16` | Тест: `0`/`false` отключает FP16 в зонде. |
| `JGPT_PROBE_MAX_BATCH` | Тест: верхняя граница перебора batch. |
| `JGPT_PROBE_MODEL` | Тест: выбор/размер тестовой модели. |
| `JGPT_LOG_COLOR` | Принудительно включить цвет в логе. |
| `NO_COLOR` | Отключить цвет в логе (стандарт). |
| `JGPT_JAVA_MEM` | Строка для `MAVEN_OPTS` (напр. `-Xmx8g`). |
| `JGPT_CHECKPOINT_ASYNC` | Асинхронная запись весов после основного чекпоинта. |

## Свойства JVM (не `JGPT_*`)

| Свойство | Назначение |
|----------|------------|
| `-Djgpt.fullGpuTrain` | Полный GPU-тренинг. |
| `-Djgpt.allow.no.gpu=true` | Разрешение работы без GPU (тесты). |
| `-Djgpt.debug.vramBeforeAlloc=true` | Лог VRAM перед крупными аллокациями. |
| `-Djgpt.decoder.graph.minFreeMib` | Минимум free VRAM перед graph-path. |
| `-Djgpt.decoder.layer.cudaGraph.log` | Логи decoder graph. |
| `-Djgpt.deviceDecoderBackward` | Decoder backward на VRAM. |
| `-Djgpt.deviceLogitsTrain` | CE и LM head backward на device. |
| `-Djgpt.fused.ffn.rms.w1w3` | Fused RMSNorm + W1/W3. |
| `-Djgpt.gpu.e2eTrain` | End-to-end GPU training. |
| `-Djgpt.sampledCe.candidates` | Кандидаты sampled CE. |
| `-Djgpt.sampledCe.negativeMode` | Режим негативов sampled CE. |
| `-Djgpt.trainLossMode` | `full` или `sampled`. |
| `-Djgpt.train.vramStepProbe` | VRAM probe вокруг decoder forward. |
| `-Djgpt.train.vramStepProbeEvery` | Интервал VRAM probe. |

Дополнительные `-Djgpt.*` перечислены в JavaDoc методов `LLMConfig` (пары к таблице выше).
