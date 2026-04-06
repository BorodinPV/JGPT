# JGPT

Обучение небольших LLM на **Java 25**: forward/backward в Java, **JNI + CUDA** (cuBLAS и собственные ядра), часть CPU-операций через **JDK Incubator Vector API**.

Целевой режим: **NVIDIA GPU**. Полноценного CPU-пути для matmul / attention / декодера нет.

- Maven: `ru.reweu:jgpt:1.0-SNAPSHOT`
- JVM: `--add-modules=jdk.incubator.vector`, `--enable-preview` (см. `pom.xml`, `.mvn/jvm.config`)
- Нативная библиотека: `./scripts/build_jgpt_cuda.sh` → `build/libjgpt_cuda.so`

## Быстрый старт

```bash
./scripts/build_jgpt_cuda.sh
./scripts/train-e2e-gpu.sh          # MultiBookTrain, e2e-дефолты
./scripts/train-e2e-gpu.sh allbooks # AllBooksTrain, весь корпус
```

Смарт-режим с пресетами: `./scripts/jgpt-smart.sh` (подробнее — `docs/training/README.md`).

## Поиск библиотеки CUDA

Порядок: `-Djgpt.cuda.lib=…` → **`JGPT_CUDA_LIB`** → `build/libjgpt_cuda.so` (подставляет `scripts/jgpt-gpu-train-lib.sh`) → `java.library.path`.

## Запуск без GPU (тесты / отладка)

- **`JGPT_ALLOW_NO_GPU=1`** или **`-Djgpt.allow.no.gpu=true`**

---

## Переменные окружения `JGPT_*`

Форматы «вкл/выкл»: обычно **`1` / `true`** = включено, **`0` / `false`** = выключено (если для конкретной переменной не сказано иное).

### Обучение: размеры и план

| Переменная | Назначение |
|------------|------------|
| **`JGPT_BATCH_SIZE`** | Переопределить размер микробатча из пресета `LLMConfig` (целое &gt; 0). |
| **`JGPT_ACCUMULATION_STEPS`** | Сколько микробатчей на один шаг оптимизатора; CE/градиенты масштабируются как `1/N` (минимум 1). |
| **`JGPT_EPOCHS`** | Число эпох поверх пресета. |
| **`JGPT_MAX_SEQ_LEN`** | Максимальная длина контекста (должна согласовываться с сохранённым чекпоинтом по позиционным эмбеддингам). |
| **`JGPT_MAX_SEQUENCES`** | В `MultiBookTrain`: ограничить число обучающих окон на книгу (альтернатива `--max-sequences`). |
| **`JGPT_EXIT_AFTER_STEP`** | После стольких шагов **оптимизатора** завершить JVM (профилирование, короткие прогоны). `0` = не ограничивать. |
| **`JGPT_FINETUNE`** | `1`/`true`: загрузить веса и Adam, сбросить `globalStep` и лучший eval (дообучение, новое LR-расписание). См. `LLMTrainer.resetGlobalStep`. |

### GPU-пайплайн и режим обучения

| Переменная | Назначение |
|------------|------------|
| **`JGPT_TRAIN_GPU_RESIDENT`** | Резидентные веса на GPU при обучении. Пусто при доступной CUDA → по сути вкл.; явный `0`/`false` выключает. |
| **`JGPT_GPU_E2E_TRAIN`** | End-to-end GPU train-пресет в `LLMConfig.toTrainingConfig`: resident + full step + device logits + device decoder (нужны CUDA и pipeline). |
| **`JGPT_FULL_GPU_TRAIN`** | Полный GPU-шаг (forward/backward/clip/Adam на device без «полупутей»). Свойство: `-Djgpt.fullGpuTrain`. |
| **`JGPT_DECODER_GPU_PIPELINE`** | Декодер слой-за-слоем на VRAM (`GPTModel`). Свойство: `-Djgpt.decoder.gpu.pipeline`. |
| **`JGPT_DECODER_LAYER_CUDA_GRAPH`** | CUDA Graph на полный слой декодера (`1`/`true` включает). Свойство: `-Djgpt.decoder.layer.cudaGraph`. |
| **`JGPT_DEVICE_LOGITS_TRAIN`** | CE и backward LM head на device. Свойство: `-Djgpt.deviceLogitsTrain`. |
| **`JGPT_DEVICE_DECODER_BWD`** | Backward блоков декодера на VRAM. Свойство: `-Djgpt.deviceDecoderBackward`. |
| **`JGPT_FUSED_LM_HEAD`** | Слияние финального RMSNorm + LM head на GPU, где поддерживается. |
| **`JGPT_FUSED_FFN_RMS_W1W3`** | Один JNI для второго RMSNorm + проекций SwiGLU W1/W3. Свойство: `-Djgpt.fused.ffn.rms.w1w3`. |
| **`JGPT_GENERATE_GPU_KV`** | Генерация с KV-кэшем на GPU (см. `LlmTextGeneration`). |
| **`JGPT_FLASH_ATTENTION`** | Экспериментальный путь FlashAttention в CUDA (если сборка и драйвер позволяют). |

### FP16 и динамический loss scale

| Переменная | Назначение |
|------------|------------|
| **`JGPT_FP16_MATMUL`** | Matmul и связанный путь в FP16; при вкл. обычно создаётся `DynamicLossScaler`. |
| **`JGPT_FP16_DYNAMIC_INITIAL`** | Начальный loss scale (float &gt; 0). |
| **`JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`** | Шагов оптимизатора между попытками увеличить scale. |
| **`JGPT_AMP_GROWTH_INTERVAL`** | То же, что интервал роста; **приоритетнее**, чем `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`. |
| **`JGPT_FP16_DYNAMIC_MAX`** | Верхняя граница scale. |
| **`JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK`** | После стольких overflow подряд на минимальном scale — сброс к начальному. |
| **`JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH`** | Сбрасывать loss scale в начале каждой эпохи. |
| **`JGPT_FP16_DYNAMIC_LOSS_SCALE`** | **Не используется** при динамическом скейлере; в логе предупреждение, если задан. |
| **`JGPT_FP16_AUX_SOFTEN`** | Глобальный выключатель вспомогательного деления scale после eval/sample (`0` = выкл.). |
| **`JGPT_FP16_AUX_SOFTEN_EVAL`** | Делитель scale после eval (по умолчанию см. код `DynamicLossScaler`). |
| **`JGPT_FP16_AUX_SOFTEN_SAMPLE`** | Делитель scale после промежуточной генерации. |

### Кэш активаций и батчи данных

| Переменная | Назначение |
|------------|------------|
| **`JGPT_ACTIVATION_CACHE_FP16`** | Хранить слоты `BlockActivationCacheDevice` в FP16 на device (где применимо). Свойство: `-Djgpt.activationCache.fp16`. |
| **`JGPT_BLOCK_CACHE_GROW_ONLY`** | Не уменьшать выделенные буферы кэша при смене batch/seq, только расти. |
| **`JGPT_BLOCK_CACHE_POOL`** | Thread-local пул кэшей блоков для повторного использования. |
| **`JGPT_BLOCK_CACHE_POOL_MAX`** | Максимум объектов в очереди пула на ключ архитектуры. |
| **`JGPT_BLOCK_CACHE_MAX_BYTES`** | Мягкий лимит оценки размера кэша в байтах (`0` = без лимита); при превышении — очистка пула и предупреждение. |
| **`JGPT_BATCH_PINNED`** | Закреплённый (pinned) хост-буфер для батча в `DataLoader`. |
| **`JGPT_BATCH_DIRECT`** | Прямой ByteBuffer для батча (off-heap). |
| **`JGPT_BATCH_PREFETCH`** | Фоновая подготовка следующего батча в `LLMTrainer`. |

### Лосс, CE, семплирование

| Переменная | Назначение |
|------------|------------|
| **`JGPT_TRAIN_LOSS_MODE`** | `full` или `sampled` (train-only; eval всегда full CE). Свойство: `-Djgpt.trainLossMode`. |
| **`JGPT_SAMPLED_CE_CANDIDATES`** | Число кандидатов на строку (target + негативы), минимум 2. Свойство: `-Djgpt.sampledCe.candidates`. |
| **`JGPT_SAMPLED_CE_NEGATIVE_MODE`** | Режим негативов (см. `SampledNegativeMode`; в коде ожидается `batch_shared_uniform`, если не задано — дефолт). Свойство: `-Djgpt.sampledCe.negativeMode`. |
| **`JGPT_CE_ASYNC`** | Асинхронный CE на GPU: очередь kernel + D2H скаляра; барьер перед backward. **Несовместимо** с `JGPT_TRAIN_LOSS_MODE=sampled` — нужен `0`. |
| **`JGPT_CE_GPU_MIN_ELEMENTS`** | Порог размера (элементов), ниже которого CE может остаться на CPU (см. `TensorOpsGPU`). |

### Чекпоинты, лог, ранний останов

| Переменная | Назначение |
|------------|------------|
| **`JGPT_CHECKPOINT_ASYNC`** | Асинхронная запись `model_*.bin` после основного чекпоинта. |
| **`JGPT_EARLY_STOP_EVAL_PATIENCE`** | Сколько eval подряд без улучшения best loss до останова; `0` = выключить эту ветку. |
| **`JGPT_EARLY_STOP_OVERFIT`** | `0`/`false` — отключить останов по признаку train↓ + eval↑. |

### Интерактив и логирование

| Переменная | Назначение |
|------------|------------|
| **`JGPT_INTERACTIVE_EVERY`** | Раз в N шагов оптимизатора — короткая генерация во время обучения; `0` = выкл. Свойство: `-Djgpt.interactiveEvery`. |
| **`JGPT_SAMPLE_PROMPT`** | Промпт(ы) для этой генерации; несколько через `|`. |
| **`JGPT_INTERACTIVE_AFTER_BOOK`** | `MultiBookTrain`: после каждой книги — интерактивная проверка в консоли (нужен TTY). |
| **`JGPT_INTERACTIVE_GEN_MAX_NEW`** | Макс. новых токенов в этой интерактивной генерации. |
| **`JGPT_INTERACTIVE_GEN_TEMP`** | Температура. |
| **`JGPT_INTERACTIVE_GEN_TOP_K`** | Top-k сэмплирования. |

### Профилирование и метрики

| Переменная | Назначение |
|------------|------------|
| **`JGPT_TRAIN_PERF`** | Расширенные строки `[PERF]` (сводка фаз, ток/с и т.д.). |
| **`JGPT_PROFILE`** | Включить `TrainingProfiler`. |
| **`JGPT_PROFILE_STEPS`** | Число шагов/глубина профиля (см. `TrainingProfiler`). |
| **`JGPT_TIMINGS`** | Краткие тайминги в логе train loss. |
| **`JGPT_STATS_JSON`** | Запись `state/stats.json` для дашборда; `0`/`false` — отключить. |
| **`JGPT_STATS_PRESET`**, **`JGPT_STATS_PRESET_IDX`** | Строки-подписи пресета в статистике (см. `LLMTrainer`). |

### Прочее (CUDA / отладка / лог)

| Переменная | Назначение |
|------------|------------|
| **`JGPT_CUDA_LIB`** | Абсолютный путь к `libjgpt_cuda.so`. |
| **`JGPT_JAVA_MEM`** | Подмешивается в `MAVEN_OPTS` скриптами обучения (например `-Xmx…`). |
| **`JGPT_RMSNORM_EPS`** | Epsilon для RMSNorm на GPU (float). |
| **`JGPT_LOG_COLOR`** | Принудительно раскрашивать префиксы лога (`LogFmt`). |
| **`JGPT_DEBUG_GPU_TRAIN`**, **`JGPT_DEBUG_GPU_TRAIN_LOG`** | Отладочный вывод GPU-обучения (см. `DebugGpuTrain`). |
| **`JGPT_BWD_LAYER_FINITE_CHECK`** | Проверки на finite после слоёв backward. |

### Только тесты / зонды (не используются в обычном `train()`)

| Переменная | Назначение |
|------------|------------|
| **`JGPT_BATCH_PROBE`**, **`JGPT_BATCH_PROBE_CPU`**, **`JGPT_BATCH_PROBE_FP16`** | Ручной зонд VRAM/батча (`EffectiveBatchProbeTest` и связанное). |
| **`JGPT_PROBE_MODEL`**, **`JGPT_PROBE_MAX_BATCH`** | Параметры зонда модели/батча. |

---

## Другие переменные окружения

| Переменная | Назначение |
|------------|------------|
| **`NO_COLOR`** | Отключить цвет в логе (`LogFmt`), общепринятый флаг. |

---

## Свойства системы `-Djgpt.*` (частичное зеркало env)

Используются там, где в коде вызывается `readBoolEnvOrProp` / `readPositiveEnvOrPropInt` / аналоги. Примеры:

- `-Djgpt.trainLossMode=sampled`
- `-Djgpt.sampledCe.candidates=128`
- `-Djgpt.decoder.gpu.pipeline=true`
- `-Djgpt.fullGpuTrain=true`
- `-Djgpt.allow.no.gpu=true`
- `-Djgpt.cuda.lib=/path/to/libjgpt_cuda.so`

Для FP16 dynamic scale свойства вида **`jgpt.fp16.dynamic.initial`**, **`jgpt.fp16.dynamic.growth.interval`** и т.д. строятся из имён env автоматически в `DynamicLossScaler` (см. исходник).

---

## Дефолты скрипта `scripts/jgpt-gpu-train-lib.sh`

При запуске через `./scripts/run-training-gpu.sh` / `train-e2e-gpu.sh` задаются разумные значения для многих `JGPT_*` (resident decoder, FP16 matmul, sampled/full и т.д.). Переменные, уже экспортированные в shell, **не перезаписываются** оператором `${VAR:-default}`.

Режим **`e2e`** дополнительно поднимает `JGPT_FULL_GPU_TRAIN=1`, `JGPT_TRAIN_PERF` (если не задан иной), смягчает `JGPT_FP16_DYNAMIC_INITIAL` и т.д. — см. функцию `jgpt_e2e_train_overrides` в `jgpt-gpu-train-lib.sh`.

---

## Реестр `System.getenv("JGPT_…")` в коде

Файл **`src/test/resources/jgpt-training-env-keys.txt`** содержит ключи, для которых в исходниках есть прямой вызов `System.getenv("JGPT_…")` (включая тесты). Сортировка и полнота проверяются тестом `JgptTrainingEnvCatalogTest`.

Переменные, читаемые только через вспомогательные функции с **динамической** строкой ключа (например `readPositiveEnvInt("JGPT_BATCH_SIZE", …)`), в этом файле **не обязаны** присутствовать, но описаны в таблицах выше.

---

## Сборка и тесты

```bash
./scripts/build_jgpt_cuda.sh
export JGPT_CUDA_LIB="$PWD/build/libjgpt_cuda.so"
mvn test
```

Surefire для большинства тестов задаёт `-Djgpt.allow.no.gpu=true`; GPU-тесты пропускаются, если CUDA недоступна.

---

## Дополнительная документация

- `docs/training/README.md` — смарт-супервизор, пресеты, чекпоинты, дашборд.
- Скрипты: `scripts/run-training-gpu.sh`, `scripts/train-e2e-gpu.sh`, `scripts/jgpt-gpu-train-lib.sh`, `scripts/build_jgpt_cuda.sh`.
