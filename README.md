# JGPT

Тензорная библиотека и обучение небольших LLM на **Java 25**: явный forward/backward, `JNI + CUDA` (cuBLAS + собственные ядра), `JDK Incubator Vector API` для части CPU-операций.

Целевой сценарий: **обучение на NVIDIA GPU**. Полноценного CPU-пути для matmul / attention / decoder нет.

Maven coordinates: `ru.reweu:jgpt:1.0-SNAPSHOT`.

---

## Требования

- `JDK 25`
- `CUDA` + `CMake`
- NVIDIA GPU (тестировалось на RTX 3080, 10 GB)
- `SLF4J + Logback` для логов

Сборка и `exec:java` используют `--add-modules=jdk.incubator.vector` и `--enable-preview` (`pom.xml`, `.mvn/jvm.config`).

---

## Быстрый старт

```bash
# Собрать нативную библиотеку
./scripts/build_jgpt_cuda.sh

# Запустить multi-book обучение (полный GPU-путь)
./scripts/train-e2e-gpu.sh

# С sampled CE (быстрее на train-loss, eval остаётся full-vocab):
JGPT_TRAIN_LOSS_MODE=sampled JGPT_CE_ASYNC=0 JGPT_SAMPLED_CE_CANDIDATES=64 \
  ./scripts/train-e2e-gpu.sh
```

Нативная библиотека ищется в таком порядке:

1. `-Djgpt.cuda.lib=...`
2. `JGPT_CUDA_LIB`
3. `build/libjgpt_cuda.so` (автоподстановка в `scripts/run-training-gpu.sh`)
4. `java.library.path`

Для окружений без GPU: `JGPT_ALLOW_NO_GPU=1` или `-Djgpt.allow.no.gpu=true`.

---

## Производительность

Тестовая конфигурация: **RTX 3080 (10 GB)**, модель `d_model=256, heads=16, layers=11`, `seq=512`, `batch=6`, sampled CE (`candidates=64`).

| Этап | До оптимизаций | После |
|---|---|---|
| forward | ~148 мс | ~28 мс |
| backward | ~302 мс | ~51 мс |
| клип+опт | ~19 мс | ~21 мс |
| **сумма/шаг** | **~470 мс** | **~100 мс** |
| **ток/с** | **~6 700** | **~31 000** |

Ускорение ~4.7×, с 6 700 до 31 000 токенов/с.

---

## CUDA-оптимизации ядер

Все изменения в `src/main/cpp/jgpt_cuda_extra.cu`:

| Оптимизация | Эффект |
|---|---|
| Softmax forward/backward: block-per-row (256 нитей на строку, warp-shuffle + shared mem) | forward −75%, backward −83% от исходных |
| RMSNorm backward: block-per-row (`rms_norm_bwd_block_kernel`) | −75% по ядру |
| RMSNorm forward: block-per-row (`rms_norm_fwd_block_kernel`, оба пути fp16/f32) | −75% по ядру |
| Транспонирование через cuBLAS `CUBLAS_OP_T`: замена 44 вызовов `transpose_last2_3d_kernel` на `batched_sgemm_row_major_transB/A` | −4% GPU-время, −буфер `K^T` в памяти |
| Fused scale+mask+softmax: `softmax_scaled_masked_block_kernel` | убирает 2 отдельных kernel launch |
| Scale поглощён в alpha GEMM (backward): убраны вызовы `scale_inplace_kernel_extra` | убирает kernel launch |
| Java: thread-local workspace-буферы, градиентный нуль-флаг, кэш `gpuTensorByTrainableParameter` | меньше `cudaMalloc`/синхронизаций |

---

## Основные точки входа

| Команда | Что делает |
|---|---|
| `./scripts/train-e2e-gpu.sh` | cmake build → `libjgpt_cuda.so` → `MultiBookTrain` (e2e GPU, PERF=1) |
| `./scripts/train-e2e-gpu.sh allbooks` | то же, но запускает `AllBooksTrain` (весь корпус как единый датасет) |
| `./scripts/run-training-gpu.sh e2e` | `MultiBookTrain` (e2e, без cmake build) |
| `./scripts/run-training-gpu.sh allbooks` | `AllBooksTrain` (без cmake build) |
| `./scripts/run-training-gpu.sh` | `MultiBookTrain` без `JGPT_FULL_GPU_TRAIN` |
| `./scripts/run-training-gpu.sh single [args…]` | `TrainLLM` |
| `./scripts/run-training-gpu.sh profile` | `ProfileQuickRun` |
| `./scripts/build_jgpt_cuda.sh` | только сборка `libjgpt_cuda.so` |
| `./run-training.sh [args…]` | алиас → `scripts/run-training-gpu.sh` |

`scripts/run-training-gpu.sh` добавляет `--enable-native-access=ALL-UNNAMED` в `MAVEN_OPTS` и, если задан `JGPT_JAVA_MEM`, подмешивает его как подсказку по heap.

---

## Как устроен запуск обучения

### `scripts/run-training-gpu.sh`

Выставляет практичные дефолты:

| Переменная | Дефолт |
|---|---|
| `JGPT_TRAIN_GPU_RESIDENT` | 1 |
| `JGPT_DECODER_GPU_PIPELINE` | 1 |
| `JGPT_DEVICE_LOGITS_TRAIN` | 1 |
| `JGPT_DEVICE_DECODER_BWD` | 1 |
| `JGPT_GPU_E2E_TRAIN` | 1 |
| `JGPT_FP16_MATMUL` | 1 |
| `JGPT_ACTIVATION_CACHE_FP16` | 1 |
| `JGPT_FUSED_LM_HEAD` | 1 |
| `JGPT_GENERATE_GPU_KV` | 1 |
| `JGPT_CE_ASYNC` | 1 (выключать при `JGPT_TRAIN_LOSS_MODE=sampled`) |
| `JGPT_TRAIN_LOSS_MODE` | `full` |
| `JGPT_SAMPLED_CE_CANDIDATES` | 128 |
| `JGPT_CHECKPOINT_ASYNC` | 0 |
| `JGPT_FULL_GPU_TRAIN` | 0 (включается префиксом `e2e`) |

### `scripts/train-e2e-gpu.sh`

Делает cmake build, затем передаёт в `run-training.sh e2e` дополнительные дефолты:

| Переменная | Дефолт |
|---|---|
| `JGPT_TRAIN_PERF` | 1 |
| `JGPT_FP16_DYNAMIC_INITIAL` | 8192 |
| `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK` | 256 |
| `JGPT_FP16_DYNAMIC_RESET_EACH EPOCH` | 0 |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | 1 |
| `JGPT_FUSED_FFN_RMS_W1W3` | 1 |

---

## Дообучение (AllBooksTrain)

`AllBooksTrain` — режим обучения, при котором все `.txt` из `data/books/` объединяются в **единый датасет** и прогоняются за один Training run. Это устраняет катастрофическое забывание, которое возникает при последовательном обучении на отдельных книгах в `MultiBookTrain`.

### Первый запуск

```bash
JGPT_TRAIN_LOSS_MODE=sampled JGPT_SAMPLED_CE_CANDIDATES=512 \
JGPT_MAX_SEQ_LEN=1024 JGPT_CE_ASYNC=0 JGPT_INTERACTIVE_EVERY=0 \
  ./scripts/train-e2e-gpu.sh allbooks
```

Чекпоинты сохраняются в `checkpoints/all_books/`, токенизатор — в `checkpoints/tokenizer_global.bin`.

### Resume после прерывания

Запустить ту же команду повторно — `AllBooksTrain` автоматически найдёт последний чекпоинт: сначала ищет `checkpoint_final.bin`, затем `checkpoint_epoch_N.bin` с максимальным N.

### Продолжение с бо́льшим числом эпох

Если обучение завершено (по умолчанию 20 эпох), но хочется продолжить:

```bash
JGPT_TRAIN_LOSS_MODE=sampled JGPT_SAMPLED_CE_CANDIDATES=512 \
JGPT_MAX_SEQ_LEN=1024 JGPT_CE_ASYNC=0 JGPT_INTERACTIVE_EVERY=0 \
JGPT_EPOCHS=40 \
  ./scripts/train-e2e-gpu.sh allbooks
```

`JGPT_EPOCHS=40` увеличивает `totalTrainingSteps`. Поскольку сохранённый `globalStep` оказывается меньше нового плана, обучение возобновляется с прерванного места.

### Дообучение на расширенном корпусе (новые книги)

Добавьте новые `.txt` в `data/books/` и запустите с флагом `JGPT_FINETUNE=1`:

```bash
JGPT_TRAIN_LOSS_MODE=sampled JGPT_SAMPLED_CE_CANDIDATES=512 \
JGPT_MAX_SEQ_LEN=1024 JGPT_CE_ASYNC=0 JGPT_INTERACTIVE_EVERY=0 \
JGPT_FINETUNE=1 JGPT_EPOCHS=20 \
  ./scripts/train-e2e-gpu.sh allbooks
```

`JGPT_FINETUNE=1` загружает веса и Adam-состояние из последнего чекпоинта, но сбрасывает `globalStep` в 0 — LR-расписание перезапускается с начала. Знания из старых книг сохраняются, Adam уже «прогрет», поэтому сходимость обычно быстрее, чем при обучении с нуля.

| Env-переменная | Описание |
|---|---|
| `JGPT_EPOCHS` | переопределить число эпох (default: 20) |
| `JGPT_FINETUNE` | `1` / `true` — сбросить `globalStep`, сохранив веса и Adam |
| `JGPT_MAX_SEQ_LEN` | переопределить длину контекста (default: 2048; для RTX 3080 рекомендуется 1024) |
| `JGPT_INTERACTIVE_EVERY` | `0` — отключить промежуточную генерацию текста во время обучения |

---



`LLMTrainer` при `useGpuResident=true` требует CUDA и корректный GPU-конвейер. Для полного GPU-шага нужны:

- `JGPT_TRAIN_GPU_RESIDENT=1`
- `JGPT_FULL_GPU_TRAIN=1`
- `JGPT_DEVICE_DECODER_BWD=1`
- `JGPT_DECODER_GPU_PIPELINE=1`
- `JGPT_DEVICE_LOGITS_TRAIN=1`
- `GPTModel.canFullGpuTrain() == true`

Что важно:

- активации decoder живут в `BlockActivationCacheDevice` на VRAM;
- host `BlockActivationCache` и `transformerBlockBackward` остаются для путей без device decoder backward;
- веса могут быть GPU-resident, но `checkpoint_*.bin` и `model_*.bin` ведутся из host-буферов через lazy sync;
- шаг обучения предполагает один поток; `GpuPendingGradients` thread-local.

### Train-only sampled CE

`JGPT_TRAIN_LOSS_MODE=sampled` включает candidate loss только в training loop:

- eval остаётся full-vocab CE — сопоставим с прежними прогонами;
- `sampled_train_loss` в train-логах служит только для мониторинга;
- LM-head считает только `rows×K` кандидатных логитов на VRAM, без материализации `rows×vocab`;
- **несовместимо** с `JGPT_CE_ASYNC=1` — при sampled-режиме выставлять `JGPT_CE_ASYNC=0`.

Рекомендуемый пресет для sampled CE:

```bash
JGPT_TRAIN_LOSS_MODE=sampled \
JGPT_CE_ASYNC=0 \
JGPT_SAMPLED_CE_CANDIDATES=64 \
  ./scripts/train-e2e-gpu.sh
```

---

## FP16 и dynamic loss scale

При `JGPT_FP16_MATMUL=1` включается `DynamicLossScaler`.

Основные переменные:

| Переменная | Описание |
|---|---|
| `JGPT_FP16_DYNAMIC_INITIAL` | начальный loss scale (default: 65536 в `run-training-gpu.sh`, 8192 в `train-e2e-gpu.sh`) |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | шагов до роста scale (default: 2000) |
| `JGPT_AMP_GROWTH_INTERVAL` | то же, приоритетнее |
| `JGPT_FP16_DYNAMIC_MAX` | максимум scale (default: 65536) |
| `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK` | overflow-серий на min → сброс к initial |
| `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH` | сбрасывать scale при смене эпохи |

`JGPT_FP16_DYNAMIC_LOSS_SCALE` в обучении **не используется**.

После eval и промежуточной генерации первый train-step может быть менее устойчивым — для сглаживания:

- `JGPT_FP16_AUX_SOFTEN_EVAL` — по умолчанию `8`
- `JGPT_FP16_AUX_SOFTEN_SAMPLE` — по умолчанию `64`
- `JGPT_FP16_AUX_SOFTEN=0` — отключить полностью

Рекомендуемый рабочий пресет:

```bash
export JGPT_FP16_DYNAMIC_INITIAL=8192
export JGPT_FP16_DYNAMIC_MAX=65536
export JGPT_FP16_DYNAMIC_GROWTH_INTERVAL=2000
export JGPT_FP16_AUX_SOFTEN_EVAL=8
export JGPT_FP16_AUX_SOFTEN_SAMPLE=64
export JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK=256
```

---

## Профилирование (Nsight Systems)

```bash
# Сделать краткий профиль (по умолчанию 8 шагов):
NSYS_OUT=/tmp/jgpt_prof ./scripts/profile-nsys.sh e2e

# Если nsys создал только .qdstrm (нет .nsys-rep):
./scripts/qdstrm-to-nsys-rep.sh /tmp/jgpt_prof.qdstrm /tmp/jgpt_prof.nsys-rep

# Экспорт в SQLite для nsys stats:
./scripts/nsys-rep-export-sqlite.sh /tmp/jgpt_prof.nsys-rep

# Топ GPU-ядер:
nsys stats /tmp/jgpt_prof.sqlite --report cuda_gpu_kern_sum
```

Важные замечания:

- **не прерывайте** `nsys profile` через Ctrl+C — `.qdstrm` без финализации обычно битый;
- `JGPT_EXIT_AFTER_STEP` управляет числом шагов до авто-остановки JVM;
- при многокнижном обучении, где все книги завершены, JVM пробегает каждую за 1-2 шага — профиль будет мультисессийным и `.qdstrm` может не импортироваться из-за разбросанных таймстемпов; в этом случае сбросьте состояние одной книги перед профилированием;
- `QdstrmImporter` ищется в `nsight-systems/host-linux-x64` (несколько стандартных путей).

---

## Ключевые `JGPT_*`

Полный алфавитный реестр всех env-переменных: `src/test/resources/jgpt-training-env-keys.txt`. Его полноту проверяет `JgptTrainingEnvCatalogTest`.

| Группа | Переменные |
|---|---|
| Native | `JGPT_CUDA_LIB`, `JGPT_ALLOW_NO_GPU` |
| GPU train path | `JGPT_TRAIN_GPU_RESIDENT`, `JGPT_GPU_E2E_TRAIN`, `JGPT_FULL_GPU_TRAIN`, `JGPT_DEVICE_LOGITS_TRAIN`, `JGPT_DEVICE_DECODER_BWD`, `JGPT_DECODER_GPU_PIPELINE`, `JGPT_DECODER_LAYER_CUDA_GRAPH`, `JGPT_FUSED_LM_HEAD`, `JGPT_FUSED_FFN_RMS_W1W3`, `JGPT_GENERATE_GPU_KV` |
| FP16 | `JGPT_FP16_MATMUL`, `JGPT_FP16_DYNAMIC_*`, `JGPT_AMP_GROWTH_INTERVAL`, `JGPT_FP16_AUX_SOFTEN*` |
| Batching / cache | `JGPT_BATCH_*`, `JGPT_BLOCK_CACHE_*`, `JGPT_ACTIVATION_CACHE_FP16` |
| CE / Loss | `JGPT_CE_ASYNC`, `JGPT_CE_GPU_MIN_ELEMENTS`, `JGPT_TRAIN_LOSS_MODE`, `JGPT_SAMPLED_CE_CANDIDATES`, `JGPT_SAMPLED_CE_NEGATIVE_MODE` |
| Perf / I/O | `JGPT_PROFILE`, `JGPT_PROFILE_STEPS`, `JGPT_TIMINGS`, `JGPT_TRAIN_PERF`, `JGPT_CHECKPOINT_ASYNC`, `JGPT_EXIT_AFTER_STEP`, `JGPT_MAX_SEQUENCES`, `JGPT_JAVA_MEM`, `JGPT_LOG_COLOR` |
| Fine-tune / конфиг | `JGPT_EPOCHS`, `JGPT_FINETUNE`, `JGPT_MAX_SEQ_LEN`, `JGPT_BATCH_SIZE`, `JGPT_INTERACTIVE_EVERY` |
| Test-only probes | `JGPT_BATCH_PROBE*`, `JGPT_PROBE_*` |

Часть флагов дублируется через `-Djgpt.*` в `LLMConfig`, `TrainingConfig` и `LLMTrainer`.

---

## Тесты

```bash
# Без GPU (большинство тестов):
mvn test

# С реальным GPU:
export JGPT_CUDA_LIB="$PWD/build/libjgpt_cuda.so"
mvn test
```

- `mvn test` запускается с `-Djgpt.allow.no.gpu=true` из Surefire;
- GPU-тесты проверяют `TensorOpsGPU.isGpuAvailable()` и пропускаются через `Assumptions`, если CUDA недоступна;
- `JGPT_BATCH_PROBE*` / `JGPT_PROBE_*` относятся к отдельным тестам, не к training launcher;
- `reuseForks=false` в Surefire уменьшает взаимное влияние JNI/CUDA между test-классами.

---

## Чекпоинты и артефакты

| Файл | Формат |
|---|---|
| `checkpoint_<name>.bin` | `veles.ckpt.v2`: шаг, лучший eval loss, буферы Adam; есть legacy-чтение через `ObjectInputStream` |
| `model_<name>.bin` | `veles.weights.v1`: формы + `float32` big-endian |
| `tokenizer_<name>.bin` | бинарный BPE-токенизатор |

При `JGPT_CHECKPOINT_ASYNC=1` запись `model_*.bin` идёт в фоне после основного `checkpoint_*.bin`.

---

## Структура кода

| Пакет | Роль |
|---|---|
| `com.veles.llm.jgpt` | `TensorOpsGPU`, `GpuFloatBuffer`, верхнеуровневые GPU-операции |
| `...core` | `Tensor` и базовые типы |
| `...model` | `GPTModel`, блоки, cache-объекты |
| `...ops` | tensor ops, backward, workspaces |
| `...training` | `LLMTrainer`, `LLMConfig`, `TrainingConfig`, `AdamOptimizer`, `DynamicLossScaler`, profiling |
| `...cuda` | `TensorCudaLibrary`, `GpuTensor`, `GpuPendingGradients` |
| `...data` | tokenizer, dataset, `DataLoader` |
| `...app` | `MultiBookTrain`, `AllBooksTrain`, `TrainLLM`, `ProfileQuickRun`, `LlmTextGeneration` |
| `...util` | общие утилиты |

---

## Полезные файлы

- `scripts/run-training-gpu.sh` — основной launcher с GPU-дефолтами
- `scripts/train-e2e-gpu.sh` — cmake build + e2e launch (рекомендуется для старта)
- `scripts/build_jgpt_cuda.sh` — только сборка `libjgpt_cuda.so`
- `scripts/profile-nsys.sh` — запись Nsight Systems профиля
- `scripts/qdstrm-to-nsys-rep.sh` — конвертация `.qdstrm` → `.nsys-rep`
- `scripts/nsys-rep-export-sqlite.sh` — экспорт `.nsys-rep` в SQLite для `nsys stats`
- `docs/ROADMAP_GPU_RESIDENT.md` — GPU-resident roadmap
- `src/test/resources/jgpt-training-env-keys.txt` — полный реестр `JGPT_*`
