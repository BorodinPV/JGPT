# Дорожная карта: GPU-резидентные веса и обучение

Цель: основная копия параметров и горячий путь forward/backward/optimizer на **VRAM**, без лишних `float[]` на каждом шаге. CPU — загрузка данных, чекпоинты, отладка.

**Текущая база:** `GpuFloatBuffer`, `GpuTensor` (`com.veles.llm.jgpt.cuda`), JNI `jgpt_cuda.cu`, `matmulGpuDevice` и др. Маршрутизация «всегда GPU при доступной CUDA» уже без старых порогов по размеру — см. `TensorOpsGPU.shouldUseGpu*`.

**Запуск:** resident + полный шаг обучения — env (`JGPT_TRAIN_GPU_RESIDENT`, `JGPT_DECODER_GPU_PIPELINE`, `JGPT_GPU_E2E_TRAIN` / `JGPT_FULL_GPU_TRAIN`); в `scripts/run-training-gpu.sh` по умолчанию уже включён E2E, префикс **`./run-training.sh e2e`** добавляет **`JGPT_FULL_GPU_TRAIN=1`** (см. `LLMConfig.toTrainingConfig`, `README.md`). При FP16 matmul используется динамический loss scale (`JGPT_FP16_DYNAMIC_INITIAL` и см. README).

---

## Этап 0 — Подготовка (инфраструктура)

| Задача | Статус |
|--------|--------|
| `GpuFloatBuffer`: `copyFromHost(FloatBuffer, …)` / `copyToHost(FloatBuffer, …)` — **direct** `FloatBuffer`, копии на потоке библиотеки (`cudaMemcpyAsync` + sync) | **Сделано** |
| `GpuTensor` — владение буфером + форма; хост для загрузки/снимка и синхронизации с чекпоинтом | **Сделано** (использование — этапы 2–5) |

---

## Этап 1 — Один слой end-to-end на GPU (Linear)

| Задача | Статус |
|--------|--------|
| `Linear` (`forwardGpu` / `backwardGpu`), `matmulGpuDevice` + `matmulGpuDeviceEx`, bias: `addBiasBroadcastGpuDevice`, градиент bias: `sumColumnsGpuDevice` | **Сделано** |
| JUnit `LinearGpuTest` vs CPU | **Сделано** |

---

## Этап 2 — GPTModel с `GpuTensor`

| Задача | Статус |
|--------|--------|
| Флаг `gpuResident` в `GPTModel`: VRAM для токен/позиционных таблиц, весов декодера (attention + FFN), финального RMSNorm и LM head; синхронизация с хостом после загрузки весов / шага оптимизатора | **Сделано** |
| `GPTModel.forwardGpuLmHead` — RMSNorm + LM head на device без H2D весов каждый шаг; `syncGpuResidentWeightsFromHost` после `loadWeights` | **Сделано** |
| Декодер на GPU | **Сделано** через `DecoderBlock.forward` + `TensorOps`; стек вынесен в `GPTModel.forwardDecoderStack`; явная точка входа блока — `DecoderBlock.forwardGpuPipeline` (обёртка над `forward` с RoPE). Отдельного `backwardGpu` на блоке нет — backward через `TransformerBackward` / `backwardDecoderLayersDevice`. |
| KV-кэш на GPU для инференса (`KvCacheGpu`, overloads `forwardPrefill` / `forwardDecode`, генерация с KV на VRAM) | **Сделано** |
| CPU `forward` / `backward` без изменений семантики при отключённом resident | **Сохранено** |

---

## Этап 3 — AdamOptimizer

| Задача | Статус |
|--------|--------|
| JNI `adamWStepGPUDevice` / `TensorOpsGPU.adamWStepGpuDevice` — то же ядро, что host-путь, без H2D/D2H | **Сделано** |
| `AdamOptimizer.stepGpu(GpuTensor, GpuTensor m, GpuTensor v)` и `stepAllGpu(List, List, List)` | **Сделано** |
| Один launch AdamW по нескольким `GpuTensor` (`adamWStepGPUDeviceFused` / блок на сегмент; без H2D ∂ и без contiguous merge) | **Сделано** |

---

## Этап 4 — LLMTrainer

| Задача | Статус |
|--------|--------|
| `TrainingConfig.useGpuResident`, `fullGpuTrainStep`, `deviceLogitsTrainStep`, `deviceDecoderBackward`; в `LLMTrainer` — согласование с `GPTModel.isGpuResident()` / `canFullGpuTrain()` | **Сделано** |
| После шага Adam: `GPTModel.onParametersUpdated()` → синхронизация VRAM с хостом по необходимости | **Сделано** |
| Forward с resident-хвостом: RMSNorm + LM head на GPU; CE на device при соответствующем конфиге; backward декодера на device — `backwardDecoderLayersDevice` | **Сделано** (при включённом pipeline и флагах конфига) |
| Полный шаг на device: `clipAndOptimizerStepFullGpu` (проверка overflow на VRAM, unscale, clip, fused Adam на `GpuTensor`) | **Сделано** |
| Батч → **pinned** host (`cudaHostAlloc`): `Tensor.allocatePinnedHost`, JNI `CudaPinnedHost`, `DataLoader` + `JGPT_BATCH_PINNED` / `-Djgpt.batch.pinned=true` (подразумевает direct-путь; освобождение через `Cleaner`) | **Сделано** |
| Монолитный вход стека декодера: `GPTModel.forwardDecoderStack`; явный вход блока: `DecoderBlock.forwardGpuPipeline` (= прежний `forward` с RoPE) | **Сделано** |
| Меньше launch’ов в attention: Q/K/V после первой RMSNorm — один `cublasSgemmStridedBatched` (упаковка `Wq|Wk|Wv`, `stride=0` для общего `X`) через `TensorOpsGPU.matmulGpuDeviceQkvProjections` вместо трёх `matmulGpuDeviceEx` на resident-пути | **Сделано** |
| Меньше launch’ов в SwiGLU FFN: `W1` и `W3` после второй RMSNorm — один `cublasSgemmStridedBatched` (`matmulGpuDeviceFfnW1W3Projections`) вместо двух `matmulGpuDeviceEx` | **Сделано** |
| Один **CUDA graph** на полный декодер-слой (MHA+FFN) на `kTensorCudaStream`: env `JGPT_DECODER_LAYER_CUDA_GRAPH=1` / `-Djgpt.decoder.layer.cudaGraph=true` (см. `TensorOpsGPU.cudaStreamBeginCapture`, `GPTModel#runDecoderStackLayers`); при ошибке захвата — откат на обычные launch’ы | **Сделано** (graph; не single-kernel) |
| **Фикс CUDA graph: `tl_attn_fwd_aux` → `tl_attn_fwd_graph_aux`** — `attn_fwd_aux_ensure_qk_probs_only` (graph-путь) и `attn_fwd_aux_ensure` (non-graph, с mask-слотом) использовали один и тот же thread-local буфер. Генерация текста вызывала `scaledDotProductAttentionForwardGPUDevice` с маской → буфер перевыделялся → GPU-адреса в захваченном графе устаревали → `cudaGraphLaunch failed: illegal memory access`. Исправлено разделением на два независимых буфера в `jgpt_cuda_extra.cu`. | **Сделано** |
| Resident SDPA без `cudaStreamSynchronize` на probs: device-маска + D2H probs убраны с горячего пути (`scaledDotProductAttentionForwardGpuDeviceResident`) | **Сделано** |
| Fusion RMS→GEMM до `W1+W3`: env `JGPT_FUSED_FFN_RMS_W1W3=1` (`rmsNormMatmulFfnW1W3GpuDevice`, один JNI vs RMS + `matmulGpuDeviceFfnW1W3Projections`) | **Сделано** |
| Дальше: **single mega-kernel** / FlashAttention | План |
| Единый контракт **скаляра CE** + fused по direct/pinned: `TensorOpsGPU.crossEntropySoftmaxGradLossGpuDirectEx` (JNI `crossEntropySoftmaxGradLossGPUDirect`); device CE: `uploadTokenIdsFromFloatDirectToGpuInt` / `copyHostFloatBufferToGpuIntTokenIds` без цикла `float[]`→`int[]` на direct target | **Сделано** |
| Resident `clipAndOptimizerStep` при `allDirtyTargetsHaveGpuTensor`: merge-first, finite на VRAM, `unscaleGpuDeviceGrads`, clip на device, один D2H перед Adam; иначе `flushAllToHost`; full-GPU шаг без `accumulateAddGpuFromHost` | **Сделано** |
| **`TrainingStatsWriter`**: атомарная запись `state/stats.json` (tmp→rename) после каждого шага, eval, сэмпла, overflow; читается `dashboard.html` (Chart.js, автообновление 30 с) | **Сделано** |
| **`SmartTrainingSupervisor`** + **`PresetDecider`** + **`TrainingEventCallback`**: вся адаптивная логика переключения пресетов перенесена из bash в Java; один JVM на всё обучение; `jgpt-smart.sh` стал тонкой (~60 строк) обёрткой | **Сделано** |

---

## Этап 5 — Данные и эмбеддинги

| Задача | Статус |
|--------|--------|
| `DataLoader`: два переиспользуемых слота `Tensor` input/target + scratch `int[]` токенов (меньше аллокаций на батч; совместимо с prefetch) | **Сделано** |
| `Tensor.allocateDirect` / **`Tensor.allocatePinnedHost`**; батч: `JGPT_BATCH_DIRECT`, **`JGPT_BATCH_PINNED`**; JNI `embeddingTokenForwardGPUDirect` (H2D с адреса direct `ByteBuffer`) | **Сделано** |
| Таблица токен-эмбеддингов на VRAM при `gpuResident`; gather без H2D весов (`embeddingTokenForwardGPU*DeviceWeights`); синхронизация в `syncGpuResidentWeightsFromHost` | **Сделано** |
| Градиенты эмбеддинга полностью на device (backward scatter на VRAM + D2H в `gradBuffer` для Adam) | **Сделано** |
| Таблица позиционных эмбеддингов на VRAM при `gpuResident`; scatter ∂ позиций на device (как у токенов) | **Сделано** |
| Forward: сложение позиций к активациям с VRAM (`addPositionEmbeddingGPUDeviceWeights`) при пороге elementwise | **Сделано** |
| Gather токенов → device-буфер + позиции на device (`*ToDevice` + `addPositionEmbeddingGPUDeviceBuffers`), один D2H в тензор | **Сделано** |

---

## Зависимости и риски

- **VRAM**: все веса + Adam + активации должны помещаться (или меньший batch / короче контекст).
- **Покрытие ядер**: каждый слой в GPU-пути должен иметь device-реализацию; при смешанном пути следить за согласованностью flush (`GpuPendingGradients`) и одним потоком обучения.
- **Синхронизация потоков CUDA** — критична для memset/memcpy (см. историю правок в `nativeClear` / `nativeCopyHtoD`).
- **FP16 на полном GPU-шаге**: возможны нечисловые градиенты на VRAM при слишком агрессивном статическом scale — см. динамический scaler и запуск через `./run-training.sh` / префикс `e2e` (`README.md`).
