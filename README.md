| Переменная | Описание |
|------------|----------|
| `JGPT_ACCUMULATION_STEPS` | Сколько микробатчей на один шаг оптимизатора; CE и градиенты масштабируются как 1/N (минимум 1). |
| `JGPT_ACTIVATION_CACHE_FP16` | Хранить слоты кэша активаций блоков на GPU в FP16 (где поддерживается). |
| `JGPT_ALLOW_NO_GPU` | Разрешить работу без CUDA (тесты, отладка). |
| `JGPT_AMP_GROWTH_INTERVAL` | Интервал шагов до роста FP16 loss scale; приоритетнее `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`. |
| `JGPT_BATCH_DIRECT` | Off-heap прямой буфер для батча в `DataLoader`. |
| `JGPT_BATCH_PINNED` | Pinned host-память для батча в `DataLoader`. |
| `JGPT_BATCH_PREFETCH` | Фоновая подготовка следующего батча в `LLMTrainer`. |
| `JGPT_BATCH_PROBE` | Тест `EffectiveBatchProbeTest`: включить зонд при значении `1`. |
| `JGPT_BATCH_PROBE_CPU` | Зонд батча: форсировать CPU-режим там, где применимо. |
| `JGPT_BATCH_PROBE_FP16` | Зонд батча: `0`/`false` отключает FP16 в зонде. |
| `JGPT_BATCH_SIZE` | Переопределить размер микробатча из пресета `LLMConfig` (целое > 0). |
| `JGPT_BLOCK_CACHE_GROW_ONLY` | Кэш активаций блоков на GPU только растёт, не сжимается при смене формы. |
| `JGPT_BLOCK_CACHE_MAX_BYTES` | Мягкий лимит оценки размера кэша в байтах; `0` — без лимита. |
| `JGPT_BLOCK_CACHE_POOL` | Thread-local пул объектов кэша блоков для повторного использования. |
| `JGPT_BLOCK_CACHE_POOL_MAX` | Максимум объектов в очереди пула на один ключ архитектуры. |
| `JGPT_BWD_LAYER_FINITE_CHECK` | Проверки finite после слоёв backward (отладка). |
| `JGPT_CE_ASYNC` | Асинхронный CE на GPU; несовместимо с `JGPT_TRAIN_LOSS_MODE=sampled` — ставить `0`. |
| `JGPT_CE_GPU_MIN_ELEMENTS` | Порог размера; ниже него CE может остаться на CPU. |
| `JGPT_CHECKPOINT_ASYNC` | Асинхронная запись `model_*.bin` после основного чекпоинта. |
| `JGPT_CUDA_LIB` | Абсолютный путь к `libjgpt_cuda.so`. |
| `JGPT_DEBUG_GPU_TRAIN` | Включить отладочный режим GPU-обучения (`DebugGpuTrain`). |
| `JGPT_DEBUG_GPU_TRAIN_LOG` | Путь/режим лога для отладки GPU-train. |
| `JGPT_DECODER_GPU_PIPELINE` | Декодер на GPU слой-за-слоем (резидентный pipeline). |
| `JGPT_DECODER_LAYER_CUDA_GRAPH` | CUDA Graph на полный слой декодера (`1`/`true` — вкл.). |
| `JGPT_DEVICE_DECODER_BWD` | Backward декодера на VRAM. |
| `JGPT_DEVICE_LOGITS_TRAIN` | CE и backward LM head на device. |
| `JGPT_EARLY_STOP_EVAL_PATIENCE` | Число eval подряд без улучшения best loss до останова; `0` — выкл. |
| `JGPT_EARLY_STOP_OVERFIT` | `0`/`false` — отключить останов по признаку train↓ и eval↑. |
| `JGPT_EPOCHS` | Число эпох поверх значения в пресете. |
| `JGPT_EXIT_AFTER_STEP` | Завершить JVM после стольких шагов оптимизатора; `0` — не ограничивать. |
| `JGPT_FINETUNE` | `1`/`true`: загрузить веса и Adam, сбросить `globalStep` и лучший eval. |
| `JGPT_FLASH_ATTENTION` | Включить путь FlashAttention в CUDA (если поддерживается сборкой). |
| `JGPT_FP16_AUX_SOFTEN` | `0` — отключить вспомогательное деление loss scale после eval/sample. |
| `JGPT_FP16_AUX_SOFTEN_EVAL` | Делитель loss scale после eval (см. `DynamicLossScaler`). |
| `JGPT_FP16_AUX_SOFTEN_SAMPLE` | Делитель loss scale после промежуточной генерации. |
| `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` | Шагов оптимизатора между попытками увеличить loss scale. |
| `JGPT_FP16_DYNAMIC_INITIAL` | Начальное значение динамического loss scale (> 0). |
| `JGPT_FP16_DYNAMIC_LOSS_SCALE` | Устарело для динамического скейлера; при задании — предупреждение в логе. |
| `JGPT_FP16_DYNAMIC_MAX` | Верхняя граница loss scale. |
| `JGPT_FP16_DYNAMIC_RECOVERY_AFTER_MIN_STREAK` | После стольких overflow на min scale — сброс к начальному. |
| `JGPT_FP16_DYNAMIC_RESET_EACH_EPOCH` | Сбрасывать loss scale в начале каждой эпохи. |
| `JGPT_FP16_MATMUL` | Matmul и связанный путь в FP16; обычно включает динамический loss scale. |
| `JGPT_FULL_GPU_TRAIN` | Полный шаг обучения на GPU без «полупутей». |
| `JGPT_FUSED_FFN_RMS_W1W3` | Слияние второго RMSNorm и проекций SwiGLU W1/W3 в один JNI-вызов. |
| `JGPT_FUSED_LM_HEAD` | Слияние финального RMSNorm и LM head на GPU, где поддерживается. |
| `JGPT_GENERATE_GPU_KV` | Генерация с KV-кэшем на GPU. |
| `JGPT_GPU_E2E_TRAIN` | Пресет end-to-end GPU в `LLMConfig.toTrainingConfig` (resident + full step + device logits/decoder). |
| `JGPT_INTERACTIVE_AFTER_BOOK` | `MultiBookTrain`: после каждой книги — интерактив в консоли (нужен TTY). |
| `JGPT_INTERACTIVE_EVERY` | Каждые N шагов оптимизатора — короткая генерация; `0` — выкл. |
| `JGPT_INTERACTIVE_GEN_MAX_NEW` | Макс. новых токенов в интерактивной генерации после книги. |
| `JGPT_INTERACTIVE_GEN_TEMP` | Температура для этой генерации. |
| `JGPT_INTERACTIVE_GEN_TOP_K` | Top-k для этой генерации. |
| `JGPT_JAVA_MEM` | Строка для подмешивания в `MAVEN_OPTS` (например `-Xmx…`) в скриптах запуска. |
| `JGPT_LOG_COLOR` | Принудительно включить цветные префиксы в логе. |
| `JGPT_MAX_SEQUENCES` | `MultiBookTrain`: лимит обучающих окон на книгу. |
| `JGPT_MAX_SEQ_LEN` | Макс. длина контекста; должна согласовываться с чекпоинтом. |
| `JGPT_PROBE_MAX_BATCH` | Зонд: верхняя граница перебора batch. |
| `JGPT_PROBE_MODEL` | Зонд: выбор/размер тестовой модели. |
| `JGPT_PROFILE` | Включить `TrainingProfiler`. |
| `JGPT_PROFILE_STEPS` | Глубина/число шагов профилирования. |
| `JGPT_RMSNORM_EPS` | Epsilon для RMSNorm на GPU. |
| `JGPT_SAMPLED_CE_CANDIDATES` | Число кандидатов на строку в sampled train loss (минимум 2). |
| `JGPT_SAMPLED_CE_NEGATIVE_MODE` | Режим негативов для sampled CE (см. `SampledNegativeMode`). |
| `JGPT_SAMPLE_PROMPT` | Промпт для промежуточной генерации; несколько через `|`. |
| `JGPT_STATS_JSON` | Запись `state/stats.json`; `0`/`false` — отключить. |
| `JGPT_STATS_PRESET` | Строка-подпись пресета в статистике. |
| `JGPT_STATS_PRESET_IDX` | Индекс/подпись пресета в статистике. |
| `JGPT_TIMINGS` | Краткие тайминги в строках train loss. |
| `JGPT_TRAIN_GPU_RESIDENT` | Резидентные веса на GPU; явный `0`/`false` выключает. |
| `JGPT_TRAIN_LOSS_MODE` | `full` или `sampled` (только train; eval всегда full CE). |
| `JGPT_TRAIN_PERF` | Расширенные строки `[PERF]` (фазы, ток/с и т.д.). |
| `NO_COLOR` | Отключить цвет в логе (общепринятый флаг). |
