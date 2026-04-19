# Frequently Asked Questions / Часто задаваемые вопросы

## Performance / Производительность

### Q: What throughput should I expect? / Какую производительность ожидать?
**A:** On RTX 3080 (10GB) with preset 02-stable:  
**Ответ:** На RTX 3080 (10GB) с пресетом 02-stable:
- ~26,000 tokens/sec / ~26,000 токенов/сек
- ~1250ms per step (forward 600ms + backward 620ms + optimizer 30ms) / ~1250мс на шаг

### Q: How does JGPT compare to PyTorch? / Как JGPT сравнивается с PyTorch?
**A:** JGPT achieves ~1.0-1.2x PyTorch performance for similar models on same hardware, due to:  
**Ответ:** JGPT достигает ~1.0-1.2x производительности PyTorch для аналогичных моделей на том же железе, благодаря:
- Custom optimized CUDA kernels / Кастомным оптимизированным CUDA ядрам
- No Python overhead / Отсутствию оверхеда Python
- Direct cuBLAS integration / Прямой интеграции cuBLAS

---

## Build Issues / Проблемы сборки

### Q: GCC 15 is not supported by CUDA / GCC 15 не поддерживается CUDA
**A:** Add `-allow-unsupported-compiler` flag to CMakeLists.txt or use GCC ≤ 13.  
**Ответ:** Добавьте флаг `-allow-unsupported-compiler` в CMakeLists.txt или используйте GCC ≤ 13.

### Q: `cudaFuncSetAttribute smem=110592: invalid argument`
**A:** FlashAttention tile size is too large for your GPU. Use smaller tile:  
**Ответ:** Размер плитки FlashAttention слишком большой для вашей GPU. Используйте меньший:
```bash
JGPT_FA_TILE_SIZE=128 cmake ..
```

**Maximum tile sizes by GPU / Максимальные размеры плитки по GPU:**
- RTX 3080/4090: 128 (82KB shared memory / shared memory)
- A100: 144-160 (110-136KB)
- H100: 192+ (184KB+)

---

## Training / Обучение

### Q: Out of memory error / Ошибка нехватки памяти
**A:** Try in order / Попробуйте по порядку:
1. Switch to preset 03-recovery or 04-minimal / Переключитесь на пресет 03-recovery или 04-minimal
2. Reduce `JGPT_BATCH_SIZE` to 1 / Уменьшите `JGPT_BATCH_SIZE` до 1
3. Reduce `JGPT_MAX_SEQ_LEN` to 512 / Уменьшите `JGPT_MAX_SEQ_LEN` до 512
4. Disable CUDA Graph: `JGPT_DECODER_LAYER_CUDA_GRAPH=0` / Отключите CUDA Graph

### Q: Training stopped with "overflow-скип" / Обучение остановилось с "overflow-скип"
**A:** FP16 scale is stuck. Training will auto-downgrade preset. You can also:  
**Ответ:** FP16 scale залип. Обучение автоматически понизит пресет. Также можно:
- Reduce `JGPT_FP16_DYNAMIC_INITIAL` / Уменьшить `JGPT_FP16_DYNAMIC_INITIAL`
- Increase `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` / Увеличить `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`

### Q: How to resume training? / Как возобновить обучение?
**A:** Just run `./scripts/jgpt-smart.sh` again. It will auto-resume from `checkpoint_final.bin`.  
**Ответ:** Просто запустите `./scripts/jgpt-smart.sh` снова. Оно автоматически возобновится из `checkpoint_final.bin`.

---

## Configuration / Конфигурация

### Q: What presets are available? / Какие пресеты доступны?
| Preset | Use case / Назначение | Batch | CUDA Graph |
|--------|----------------------|-------|------------|
| 00-max-throughput | Maximum speed / Максимальная скорость | 4 | Off (OOM risk / риск OOM) |
| 01-aggressive | Default start / Старт по умолчанию | 1 | On |
| 02-stable | Stable training / Стабильное обучение | 2 | On |
| 03-recovery | After OOM / После OOM | 1 | Off |
| 04-minimal | Last resort / Последний вариант | 1 | Off |

### Q: How to change FlashAttention tile size? / Как изменить размер плитки FlashAttention?
**A:** Tile size is compile-time constant. Rebuild with:  
**Ответ:** Размер плитки - константа времени компиляции. Пересоберите с:
```bash
cd build
JGPT_FA_TILE_SIZE=128 cmake ../src/main/cpp
cmake --build .
```

Valid values / Допустимые значения: 64, 96, 128, 144 (A100+), 160 (A100+), 192 (H100+)

---

## Architecture / Архитектура

### Q: Why is backward pass slower than forward? / Почему backward медленнее forward?
**A:** Normal for transformers. Backward requires:  
**Ответ:** Нормально для трансформеров. Backward требует:
- Gradient computation for all parameters / Вычисления градиентов для всех параметров
- Weight updates / Обновления весов
- More memory bandwidth / Больше памяти bandwidth

Typical ratio / Типичное соотношение: backward = 1.0-1.2x forward time.

### Q: What optimizations are implemented? / Какие оптимизации реализованы?
- FlashAttention-2 (fused attention / слитое внимание)
- Block-per-row Cross-Entropy (12x faster / в 12 раз быстрее)
- Warp-level reduction for embeddings / Редукция уровня warp для embeddings
- FP16 Tensor Cores for GEMM / FP16 Tensor Cores для GEMM
- CUDA Graph for decoder layers / CUDA Graph для слоёв декодера
- Async checkpointing / Асинхронное сохранение чекпоинтов
