# JGPT — переменные окружения

Список **`JGPT_*`**, которые читает код (через `System.getenv` или обёртки вроде `readPositiveEnvInt("JGPT_…", …)`). Скрипт **`./scripts/jgpt-smart.sh`** подмешивает файлы `env/<пресет>.env` в окружение процесса перед Maven — JVM наследует те же переменные.

Многие флаги имеют **парные system properties `jgpt.*`** (например `JGPT_FULL_GPU_TRAIN` ↔ `-Djgpt.fullGpuTrain`): если в описании не указано иное, см. JavaDoc в `LLMConfig` и соответствующих классах.

Полный перечень литеральных `System.getenv("JGPT_…")` в исходниках **и** тестах поддерживается в `src/test/resources/jgpt-training-env-keys.txt` (тест `JgptTrainingEnvCatalogTest`).

| Переменная | Описание |
|------------|----------|
| `JGPT_ACCUMULATION_STEPS` | Сколько микробатчей градиента на один шаг оптимизатора; CE и градиенты масштабируются как 1/N (минимум 1). См. `LLMConfig.applyAccumulationStepsOverrideFromEnv`. |
| `JGPT_ACTIVATION_CACHE_FP16` | Хранить слоты кэша активаций блоков на GPU в FP16 (где поддерживается). |
| `JGPT_ALLOW_NO_GPU` | Разрешить работу без CUDA (тесты, отладка; в Maven Surefire часто эквивалентно `-Djgpt.allow.no.gpu=true`). |
| `JGPT_AMP_GROWTH_INTERVAL` | Интервал шагов до роста FP16 loss scale; приоритетнее `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`. См. `DynamicLossScaler.fromEnvironmentIfFp16`. |
| `JGPT_BATCH_DIRECT` | Off-heap прямой буфер для батча в `DataLoader`. |
| `JGPT_BATCH_PINNED` | Pinned host-память для батча в `DataLoader`. |
| `JGPT_BATCH_PREFETCH` | Фоновая подготовка следующего батча в `LLMTrainer`. |
| `JGPT_BATCH_PROBE` | Только тесты: при `1` включает opt-in сценарий в `EffectiveBatchProbeTest` (подбор batch/accum). В `train()` не читается. |
| `JGPT_BATCH_PROBE_CPU` | Тест `EffectiveBatchProbeTest`: форсировать CPU там, где применимо. |
| `JGPT_BATCH_PROBE_FP16` | Тест `EffectiveBatchProbeTest`: `0`/`false` отключает FP16 в зонде. |
| `JGPT_BATCH_SIZE` | Переопределить размер микробатча в `LLMConfig` (целое > 0). |
| `JGPT_BLOCK_CACHE_GROW_ONLY` | Кэш активаций блоков на GPU только растёт, не сжимается при смене формы. |
| `JGPT_BLOCK_CACHE_MAX_BYTES` | Мягкий лимит оценки размера кэша в байтах; `0` — без лимита. |
| `JGPT_BLOCK_CACHE_POOL` | Thread-local пул объектов кэша блоков для повторного использования. |
| `JGPT_BLOCK_CACHE_POOL_MAX` | Максимум объектов в очереди пула на один ключ архитектуры. |
| `JGPT_BWD_LAYER_FINITE_CHECK` | Проверки finite после слоёв backward (отладка). |
| `JGPT_CE_ASYNC` | Асинхронный CE на GPU; с `JGPT_TRAIN_LOSS_MODE=sampled` обычно ставят `0`. |
| `JGPT_CE_GPU_MIN_ELEMENTS` | Порог размера; ниже него CE может остаться на CPU. |
| `JGPT_CHECKPOINT_ASYNC` | Асинхронная запись весов после основного чекпоинта (`model_*.bin`). |
| `JGPT_CUDA_LIB` | Абсолютный путь к `libjgpt_cuda.so`. |
| `JGPT_DEBUG_CURSOR_B39372` | Расширенный pre-launch-лог для отладки CUDA graph / cursor (см. `CursorDebugB39372`). |
| `JGPT_DEBUG_GPU_TRAIN` | Включить отладочный режим GPU-обучения (`DebugGpuTrain`, JSONL/события). |
| `JGPT_DEBUG_GPU_TRAIN_LOG` | Путь или режим файла лога для `JGPT_DEBUG_GPU_TRAIN`. |
| `JGPT_DECODER_CUDA_GRAPH_MEM_LOG` | Лог VRAM после graph launch слоя: env / `-Djgpt.decoder.cudaGraph.memLog`. |
| `JGPT_DECODER_GPU_PIPELINE` | Декодер на GPU слой-за-слоем (резидентный pipeline); нужен для полного GPU train. |
| `JGPT_DECODER_GRAPH_MIN_FREE_MIB` | Минимум `cudaMemGetInfo` free (МиБ) перед graph-path слоя; `0` — выкл. Также `jgpt.decoder.graph.minFreeMib`. |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | CUDA Graph на полный слой декодера (`1`/`true` — вкл.). |
| `JGPT_DECODER_LAYER_CUDA_GRAPH_LOG` | Подробные логи указателей decoder graph. Свойство: `jgpt.decoder.layer.cudaGraph.log`. |
| `JGPT_DEVICE_DECODER_BWD` | Backward декодера на VRAM. Свойство: `jgpt.deviceDecoderBackward`. |
| `JGPT_DEVICE_LOGITS_TRAIN` | CE и backward LM head на device. Свойство: `jgpt.deviceLogitsTrain`. |
| `JGPT_EARLY_STOP_EVAL_PATIENCE` | Число eval подряд без улучшения best loss до останова; `0` — выкл. |
| `JGPT_EARLY_STOP_OVERFIT` | `0`/`false` — отключить останов по признаку train↓ и eval↑. |
| `JGPT_EPOCHS` | Число эпох поверх значения в `LLMConfig`. |
| `JGPT_EXIT_AFTER_STEP` | Завершить JVM после стольких **шагов оптимизатора**; `0` — не ограничивать. |
| `JGPT_FINETUNE` | `1`/`true`: загрузить веса и Adam, сбросить `globalStep` (дообучение с начала эпох). |
| `JGPT_FLASH_ATTENTION` | Включить путь FlashAttention в CUDA (если поддерживается сборкой). |
| `JGPT_FP16_AUX_SOFTEN` | Мастер-флаг: `0` отключает деление loss scale после вспомогательной GPU-работы (eval/генерация). |
| `JGPT_FP16_AUX_SOFTEN_EVAL` | Делитель loss scale после eval (см. `DynamicLossScaler`). |
| `JGPT_FP16_AUX_SOFTEN_SAMPLE` | Делитель loss scale после промежуточной генерации. |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | Шагов оптимизатора между попытками увеличить loss scale. |
| `JGPT_FP16_DYNAMIC_INITIAL` | Начальное значение динамического loss scale (> 0). |
| `JGPT_FP16_DYNAMIC_LOSS_SCALE` | Устарело: при задании — предупреждение; при FP16 matmul используется динамический скейлер (`_INITIAL` / `_GROWTH_INTERVAL` / `_MAX`). |
| `JGPT_FP16_DYNAMIC_MAX` | Верхняя граница loss scale. |
| `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK` | После стольких overflow подряд на min scale — сброс к начальному. |
| `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH` | Сбрасывать loss scale в начале каждой эпохи. |
| `JGPT_FP16_MATMUL` | Matmul и связанный путь в FP16; обычно включает динамический loss scale. |
| `JGPT_FULL_GPU_TRAIN` | Полный шаг обучения на GPU без «полупутей». Свойство: `jgpt.fullGpuTrain`. |
| `JGPT_FUSED_FFN_RMS_W1W3` | Один JNI для второго RMSNorm + проекций SwiGLU W1/W3. Свойство: `jgpt.fused.ffn.rms.w1w3`. |
| `JGPT_FUSED_LM_HEAD` | Слияние финального RMSNorm и LM head на GPU, где поддерживается. |
| `JGPT_GENERATE_GPU_KV` | Генерация с KV-кэшем на GPU (`LlmTextGeneration` и связанные пути). |
| `JGPT_GPU_E2E_TRAIN` | Пресет end-to-end GPU в `LLMConfig.toTrainingConfig` (resident + full step + device logits/decoder). Свойство: `jgpt.gpu.e2eTrain`. |
| `JGPT_INTERACTIVE_EVERY` | Каждые N шагов оптимизатора — короткая генерация; `0` или отрицательное — выкл. |
| `JGPT_JAVA_MEM` | Строка для подмешивания в `MAVEN_OPTS` в скриптах (например `-Xmx…`). |
| `JGPT_LOG_COLOR` | Принудительно включить цветные префиксы в логе. |
| `JGPT_MAX_SEQ_LEN` | Макс. длина контекста; должна согласовываться с чекпоинтом. |
| `JGPT_PRESET_NUM_LAYERS` | Переопределить число декодер-слоёв в `LLMConfig` (целое > 0); удобно из `env/*.env` для `AllBooksTrain` / `jgpt-smart.sh`. |
| `JGPT_PROBE_MAX_BATCH` | Тест `EffectiveBatchProbeTest`: верхняя граница перебора batch. |
| `JGPT_PROBE_MODEL` | Тест `EffectiveBatchProbeTest`: выбор/размер тестовой модели. |
| `JGPT_PROFILE` | Включить `TrainingProfiler`. |
| `JGPT_PROFILE_STEPS` | Глубина/число шагов профилирования. |
| `JGPT_RMSNORM_EPS` | Epsilon для RMSNorm на GPU. |
| `JGPT_SAMPLED_CE_CANDIDATES` | Число кандидатов на строку в sampled train loss (минимум 2). Свойство: `jgpt.sampledCe.candidates`. |
| `JGPT_SAMPLED_CE_NEGATIVE_MODE` | Режим негативов для sampled CE (`SampledNegativeMode`). Свойство: `jgpt.sampledCe.negativeMode`. |
| `JGPT_SAMPLE_PROMPT` | Промпт для промежуточной генерации; несколько вариантов через `|`. |
| `JGPT_STATS_JSON` | Запись `state/stats.json`; `0`/`false` — отключить. |
| `JGPT_STATS_PRESET` | Строка-подпись пресета в статистике (подставляет `jgpt-smart.sh`). |
| `JGPT_STATS_PRESET_IDX` | Индекс пресета в статистике. |
| `JGPT_TIMINGS` | Краткие тайминги в строках train loss. |
| `JGPT_TRAIN_GPU_RESIDENT` | Резидентные веса на GPU; явный `0`/`false` выключает. |
| `JGPT_TRAIN_LOSS_MODE` | `full` или `sampled` (train; eval — full CE). Свойство: `jgpt.trainLossMode`. |
| `JGPT_TRAIN_PERF` | Расширенные строки `[PERF]` (фазы, ток/с и т.д.). |
| `JGPT_TRAIN_VRAM_STEP_PROBE` | NDJSON-снимки VRAM вокруг decoder forward (отладка). Свойство: `jgpt.train.vramStepProbe`. |
| `JGPT_TRAIN_VRAM_STEP_PROBE_EVERY` | Интервал вызовов `forwardGpuDecoder` для пробника. Свойство: `jgpt.train.vramStepProbeEvery` (по умолчанию 50). |
| `NO_COLOR` | Отключить цвет в логе (общепринятый флаг; читается в `LogFmt`). |

## Свойства JVM (не `JGPT_*`)

| Свойство | Назначение |
|----------|------------|
| `-Djgpt.debug.vramBeforeAlloc=true` | Лог свободной VRAM перед крупными device-аллокациями (см. `TensorOpsGPU.logVramBeforeDeviceFloatAlloc`). |
| `-Djgpt.allow.no.gpu=true` | Аналог разрешения работы без GPU в тестах (рядом с `JGPT_ALLOW_NO_GPU`). |

Дополнительные `-Djgpt.*` перечислены в JavaDoc у методов `LLMConfig` (пары к таблице выше).
