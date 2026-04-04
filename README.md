# JGPT

Тензорная библиотека и обучение небольших LLM на `Java 25`: явный forward/backward, `JNI + CUDA` (cuBLAS + собственные ядра), `JDK Incubator Vector API` для части CPU-операций.

Целевой сценарий: **обучение на NVIDIA GPU**. Полноценного CPU-пути для matmul / attention / decoder нет.

Maven coordinates: `ru.reweu:jgpt:1.0-SNAPSHOT`.

## Требования

- `JDK 25`
- `CUDA` + `CMake`
- NVIDIA GPU для реального обучения
- `SLF4J + Logback` для логов

Сборка и `exec:java` используют `--add-modules=jdk.incubator.vector` и `--enable-preview` (`pom.xml`, `.mvn/jvm.config`).

## Быстрый старт

```bash
cd src/main/cpp
cmake -B ../../build -S .
cmake --build ../../build
cd ../..

mvn -q compile
mvn -q test
```

Нативная библиотека ищется в таком порядке:

1. `-Djgpt.cuda.lib=...`
2. `JGPT_CUDA_LIB`
3. `build/libjgpt_cuda.so`
4. `java.library.path`

Для окружений без GPU есть обход `JGPT_ALLOW_NO_GPU=1` или `-Djgpt.allow.no.gpu=true`, но для реального обучения их задавать не нужно.

## Основные точки входа

По умолчанию `mvn exec:java` запускает `com.veles.llm.jgpt.app.MultiBookTrain`.

| Команда | Что делает |
|---|---|
| `./run-training.sh` | `MultiBookTrain` с env из `scripts/run-training-gpu.sh` |
| `./run-training.sh e2e` | То же, но ещё включает `JGPT_FULL_GPU_TRAIN=1` |
| `./run-training.sh single ...` | `TrainLLM` |
| `./run-training.sh train ...` | То же, что `single` |
| `./run-training.sh profile ...` | `ProfileQuickRun` |
| `./scripts/train-e2e-gpu.sh` | Сначала собирает `libjgpt_cuda.so`, затем запускает `./run-training.sh e2e` |
| `./scripts/build_jgpt_cuda.sh` | Только сборка нативной библиотеки |

`./run-training.sh` просто делегирует в `scripts/run-training-gpu.sh`.

`scripts/run-training-gpu.sh` также добавляет `--enable-native-access=ALL-UNNAMED` в `MAVEN_OPTS` и, если задан `JGPT_JAVA_MEM`, подмешивает его туда же как подсказку по heap для JVM Maven.

## Typical workflows

```bash
# 1. Собрать CUDA-библиотеку и проверить проект
./scripts/build_jgpt_cuda.sh
mvn -q test

# 2. Запустить multi-book обучение с типовыми GPU-настройками
./run-training.sh

# 3. Запустить полный GPU e2e-сценарий с предварительной сборкой .so
./scripts/train-e2e-gpu.sh
```

## Как устроен запуск обучения

`scripts/run-training-gpu.sh` выставляет практичные дефолты для обучения:

- `JGPT_TRAIN_GPU_RESIDENT=1`
- `JGPT_DECODER_GPU_PIPELINE=1`
- `JGPT_DEVICE_LOGITS_TRAIN=1`
- `JGPT_DEVICE_DECODER_BWD=1`
- `JGPT_GPU_E2E_TRAIN=1`
- `JGPT_FP16_MATMUL=1`
- `JGPT_CE_ASYNC=1`
- `JGPT_ACTIVATION_CACHE_FP16=1`
- `JGPT_FUSED_LM_HEAD=1`
- `JGPT_GENERATE_GPU_KV=1`
- `JGPT_CHECKPOINT_ASYNC=0`
- `JGPT_TRAIN_LOSS_MODE=full`
- `JGPT_SAMPLED_CE_CANDIDATES=128`
- `JGPT_SAMPLED_CE_NEGATIVE_MODE=batch_shared_uniform`

При префиксе `e2e` скрипт дополнительно включает `JGPT_FULL_GPU_TRAIN=1`.

`scripts/train-e2e-gpu.sh` задаёт более консервативный FP16-пресет для полного GPU-цикла:

- `JGPT_FP16_DYNAMIC_INITIAL=8192`
- `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK=256`
- `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH=0`
- `JGPT_TRAIN_PERF=1`
- `JGPT_DECODER_LAYER_CUDA_GRAPH=1`
- `JGPT_FUSED_FFN_RMS_W1W3=1`

## GPU-путь и ограничения

`LLMTrainer` при `useGpuResident=true` требует CUDA и корректный GPU-конвейер модели. Для полного GPU-шага нужны:

- `JGPT_TRAIN_GPU_RESIDENT=1`
- `JGPT_FULL_GPU_TRAIN=1`
- `JGPT_DEVICE_DECODER_BWD=1`
- `JGPT_DECODER_GPU_PIPELINE=1`
- `JGPT_DEVICE_LOGITS_TRAIN=1`
- `GPTModel.canFullGpuTrain() == true`

Train-only sampled CE:

- `JGPT_TRAIN_LOSS_MODE=sampled` включает candidate loss только в training loop.
- Eval остаётся full-vocab CE, поэтому `loss` на eval сопоставим с прежними прогонами, а `sampled_train_loss` в train-логах служит только для мониторинга train-path.
- Первая реализация поддерживается только на unified full-GPU path (`fullGpuTrainStep + deviceLogitsTrainStep + deviceDecoderBackward`).
- Пока не поддерживается вместе с `JGPT_CE_ASYNC=1`.

Что важно:

- активации decoder на этом пути живут в `BlockActivationCacheDevice` на VRAM;
- host `BlockActivationCache` и `transformerBlockBackward` остаются для путей без device decoder backward;
- веса могут быть GPU-resident, но `checkpoint_*.bin` и `model_*.bin` ведутся из host-буферов через lazy sync;
- шаг обучения предполагает один поток; `GpuPendingGradients` thread-local.

## FP16 и dynamic loss scale

При `JGPT_FP16_MATMUL=1` включается `DynamicLossScaler`.

Основные переменные:

- `JGPT_FP16_DYNAMIC_INITIAL`
- `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`
- `JGPT_AMP_GROWTH_INTERVAL` — приоритетнее, чем `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`
- `JGPT_FP16_DYNAMIC_MAX`
- `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK`
- `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH`

`JGPT_FP16_DYNAMIC_LOSS_SCALE` в обучении **не используется**.

После `eval` и промежуточной генерации первый train-step может быть менее устойчивым, поэтому есть точечное снижение текущего scale:

- `JGPT_FP16_AUX_SOFTEN_EVAL` — по умолчанию `8`
- `JGPT_FP16_AUX_SOFTEN_SAMPLE` — по умолчанию `64`
- `JGPT_FP16_AUX_SOFTEN=0` — полностью отключить soften

Рекомендуемый рабочий пресет:

```bash
export JGPT_FP16_DYNAMIC_INITIAL=8192
export JGPT_FP16_DYNAMIC_MAX=65536
export JGPT_FP16_DYNAMIC_GROWTH_INTERVAL=2000
export JGPT_FP16_AUX_SOFTEN_EVAL=8
export JGPT_FP16_AUX_SOFTEN_SAMPLE=64
export JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK=256
```

## Ключевые `JGPT_*`

Полный алфавитный реестр всех env-переменных: `src/test/resources/jgpt-training-env-keys.txt`. Его полноту проверяет `JgptTrainingEnvCatalogTest`.

Самые важные группы:

| Группа | Переменные |
|---|---|
| Native | `JGPT_CUDA_LIB`, `JGPT_ALLOW_NO_GPU` |
| GPU train path | `JGPT_TRAIN_GPU_RESIDENT`, `JGPT_GPU_E2E_TRAIN`, `JGPT_FULL_GPU_TRAIN`, `JGPT_DEVICE_LOGITS_TRAIN`, `JGPT_DEVICE_DECODER_BWD`, `JGPT_DECODER_GPU_PIPELINE`, `JGPT_DECODER_LAYER_CUDA_GRAPH`, `JGPT_FUSED_LM_HEAD`, `JGPT_FUSED_FFN_RMS_W1W3`, `JGPT_GENERATE_GPU_KV` |
| FP16 | `JGPT_FP16_MATMUL`, `JGPT_FP16_DYNAMIC_*`, `JGPT_AMP_GROWTH_INTERVAL`, `JGPT_FP16_AUX_SOFTEN*` |
| Batching / cache | `JGPT_BATCH_*`, `JGPT_BLOCK_CACHE_*`, `JGPT_ACTIVATION_CACHE_FP16` |
| CE / perf / I/O | `JGPT_CE_ASYNC`, `JGPT_CE_GPU_MIN_ELEMENTS`, `JGPT_TRAIN_LOSS_MODE`, `JGPT_SAMPLED_CE_CANDIDATES`, `JGPT_SAMPLED_CE_NEGATIVE_MODE`, `JGPT_PROFILE`, `JGPT_PROFILE_STEPS`, `JGPT_TIMINGS`, `JGPT_TRAIN_PERF`, `JGPT_CHECKPOINT_ASYNC`, `JGPT_EXIT_AFTER_STEP`, `JGPT_MAX_SEQUENCES`, `JGPT_JAVA_MEM`, `JGPT_LOG_COLOR` |
| Test-only probes | `JGPT_BATCH_PROBE*`, `JGPT_PROBE_*` |

Часть флагов дублируется через `-Djgpt.*` в `LLMConfig`, `TrainingConfig` и `LLMTrainer`.

## Тесты

`mvn test` запускается с `-Djgpt.allow.no.gpu=true` из Surefire, поэтому большая часть тестов может грузиться без CUDA.

Если нужны реальные GPU-ветки:

```bash
export JGPT_CUDA_LIB="$PWD/build/libjgpt_cuda.so"
mvn -q test
```

Важно:

- GPU-тесты обычно проверяют `TensorOpsGPU.isGpuAvailable()` и пропускаются через `Assumptions`, если CUDA недоступна;
- probe-переменные (`JGPT_BATCH_PROBE*`, `JGPT_PROBE_*`) относятся к отдельным тестам, а не к обычному training launcher;
- `reuseForks=false` в Surefire уменьшает взаимное влияние JNI/CUDA между test-классами.

## Чекпоинты и артефакты

| Файл | Формат |
|---|---|
| `checkpoint_<name>.bin` | `veles.ckpt.v2`: шаг, лучший eval loss, буферы Adam; есть legacy-чтение через `ObjectInputStream` |
| `model_<name>.bin` | `veles.weights.v1`: формы + `float32` big-endian |
| `tokenizer_<name>.bin` | бинарный BPE-токенизатор |

При `JGPT_CHECKPOINT_ASYNC=1` запись `model_*.bin` идёт в фоне после основного `checkpoint_*.bin`.

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
| `...app` | `MultiBookTrain`, `TrainLLM`, `ProfileQuickRun`, `LlmTextGeneration` |
| `...util` | общие утилиты |

## Полезные файлы

- `scripts/run-training-gpu.sh` — основной launcher
- `scripts/train-e2e-gpu.sh` — build + e2e launch
- `scripts/build_jgpt_cuda.sh` — сборка `libjgpt_cuda.so`
- `docs/ROADMAP_GPU_RESIDENT.md` — GPU-resident roadmap
- `src/test/resources/jgpt-training-env-keys.txt` — полный реестр `JGPT_*`
