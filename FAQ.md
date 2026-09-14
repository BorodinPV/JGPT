# Frequently Asked Questions / Часто задаваемые вопросы

## Performance / Производительность

### Q: What throughput should I expect? / Какую производительность ожидать?
**A:** On RTX 3080 (10GB):  
**Ответ:** На RTX 3080 (10GB):
- 28L-wide (~134M, seq 1024, full CE, dropout): ~12,000 tokens/sec, step ≈ 5.3 s for 65,536 tokens (forward 1.5 s + CE 0.05 s + backward 3.7 s + Adam 0.2 s) / ~12k токенов/сек
- 12L canonical, preset 02-stable: ~26,000 tokens/sec, ~1250 ms per step / ~26k токенов/сек
- The same 28L-wide with sampled CE ran at ~6,000: the CPU candidate prep and the gather LM head are slower than one full GEMM / sampled CE на той же модели давал ~6k — медленнее полного GEMM

### Q: Why is full-vocab CE the default now? / Почему теперь полный CE по словарю?
**A:** Sampled CE with 384 uniform negatives out of 16k almost never penalises plausible-but-wrong tokens, so the model learned style but not facts. Full CE is both the correct objective and faster on this GPU. `JGPT_TRAIN_LOSS_MODE=full` in the wide presets; `sampled` remains for the legacy 37L/smart presets.  
**Ответ:** Sampled CE (384 равномерных негативов из 16k) почти не штрафует правдоподобные неверные токены — модель учила стиль, а не факты. Полный CE и корректнее, и быстрее. В wide-пресетах `JGPT_TRAIN_LOSS_MODE=full`; `sampled` остался в старых пресетах.

### Q: How does JGPT compare to PyTorch? / Как JGPT сравнивается с PyTorch?
**A:** JGPT achieves ~1.0-1.2x PyTorch performance for similar models on same hardware, due to:  
**Ответ:** JGPT достигает ~1.0-1.2x производительности PyTorch для аналогичных моделей на том же железе, благодаря:
- Custom optimized CUDA kernels / Кастомным оптимизированным CUDA ядрам
- No Python overhead / Отсутствию оверхеда Python
- Direct cuBLAS integration / Прямой интеграции cuBLAS

---

## Build Issues / Проблемы сборки

### Q: `cmake` is not recognized / Имя `cmake` не распознано
**A:** CMake is not on PATH. `nvcc` is only the CUDA compiler; JGPT still needs CMake + a host C++ compiler to produce `libjgpt_cuda.so` / `jgpt_cuda.dll`.  
**Ответ:** CMake не в PATH. `nvcc` — только компилятор CUDA; JNI-библиотеку всё равно собирают CMake и host-компилятор.

Windows:
```powershell
winget install Kitware.CMake
winget install Apache.Maven
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
# new PowerShell:
.\scripts\windows\build-cuda.ps1
. .\build\jgpt-cuda-env.ps1
```

If `mvn` is still unknown, IntelliJ already has Maven:
`C:\Program Files\JetBrains\IntelliJ IDEA 2026.2.0.1\plugins\maven-plugin\lib\maven3\bin\mvn.cmd`

Or run tests from the Maven tool window. Open a **new** PowerShell after winget so PATH updates.

### Q: `No CUDA toolset found` (Visual Studio / CMake)
**A:** The Visual Studio generator needs CUDA MSBuild integration (`.props` in Build Tools). That is often missing if CUDA was installed before Build Tools. `scripts/windows/build-cuda.ps1` uses **Ninja + nvcc + cl.exe** and does not need that integration. Re-run it; do not call `.\build\jgpt-cuda-env.ps1` until the script prints `OK: ...\jgpt_cuda.dll`.  
**Ответ:** Генератор Visual Studio ищет CUDA-тулсет в MSBuild, а не `nvcc`. Скрипт собирает через Ninja. `jgpt-cuda-env.ps1` появляется только после успешной сборки DLL.

Linux:
```bash
sudo apt install cmake build-essential
./scripts/linux/build-cuda.sh
```

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
**A:** FP16 scale is stuck. With `jgpt-smart.sh` the preset auto-downgrades. You can also:  
**Ответ:** FP16 scale залип. Под `jgpt-smart.sh` пресет понизится сам. Также можно:
- Reduce `JGPT_FP16_DYNAMIC_INITIAL` / Уменьшить `JGPT_FP16_DYNAMIC_INITIAL`
- Increase `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL` / Увеличить `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`

### Q: The log shows `[FP16] scale … ÷64 после генерации` and the scale drops to 1 / Scale падает до 1 после генерации
**A:** `JGPT_INTERACTIVE_EVERY>0` generates a sample every N steps, and each sample divides the loss scale by 64 while growth is only ×2 per 50 steps. Set `JGPT_INTERACTIVE_EVERY=0` (the wide presets do) and check quality with `jgpt-chat-*.cmd` or the GUI chat instead. A healthy run oscillates between 32768 and 65536 with only `÷2 после eval`.  
**Ответ:** Каждый промежуточный сэмпл делит scale на 64, рост — только ×2 за 50 шагов. Поставьте `JGPT_INTERACTIVE_EVERY=0` (в wide-пресетах уже так) и смотрите качество через чат. Здоровый прогон — scale 32768↔65536 и только `÷2 после eval`.

### Q: How to stop training? / Как остановить обучение?
**A:** Windows: `.\scripts\windows\jgpt-stop-train.cmd` (creates `state\STOP`; the trainer writes `checkpoint_final.bin` and exits) or the "Стоп" button in the GUI. Do **not** press Ctrl+C in PowerShell — it kills java without a checkpoint. Linux: Ctrl+C / SIGTERM go through the shutdown hook.  
**Ответ:** Windows — `jgpt-stop-train.cmd` или «Стоп» в GUI; Ctrl+C убивает java без чекпоинта. Linux — Ctrl+C работает через shutdown hook.

### Q: How to resume training? / Как возобновить обучение?
**A:** Run the same launcher again with no flags. The trainer loads the checkpoint with the largest `globalStep` among `checkpoint_final / step_N / epoch_N / best` (plus its paired `model_*.bin`), so resume also works after a hard kill — you lose at most `JGPT_SAVE_EVERY_STEPS` steps. `--restart-plan` keeps weights + Adam but resets step / LR schedule / epoch / best (use once after changing data or preset). `--fresh` archives the checkpoint dir and starts over.  
**Ответ:** Тот же скрипт без флагов — подхватит самый свежий чекпоинт из `final / step_N / epoch_N / best`; после жёсткого обрыва теряется не больше `JGPT_SAVE_EVERY_STEPS` шагов. `--restart-plan` — веса и Adam остаются, шаг/LR/эпоха/best с нуля. `--fresh` — архив и с нуля.

### Q: The pretrain model writes wiki-style nonsense in chat / Претрейн в чате пишет вики-бред
**A:** Expected. The pretrain corpus is ruwiki + a few classics; the model learned to continue text, not to answer. Test it with `--raw --temperature 0 --top-k 1` to see coherent completions, then run SFT (`jgpt-train-28L-wide-sft.cmd`) — that is what teaches the `<user>`/`<assistant>` format and short answers.  
**Ответ:** Ожидаемо: корпус — вики, модель продолжает текст, а не отвечает. Проверяйте `--raw --temperature 0 --top-k 1`, а формат ответов даёт SFT.

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
- FlashAttention-2 via cuDNN SDPA (fused attention / слитое внимание)
- Full-vocab CE and ∂CE on device (полный CE на GPU)
- Warp-level reduction for embeddings / Редукция уровня warp для embeddings
- FP16 Tensor Cores for GEMM / FP16 Tensor Cores для GEMM
- GPU dropout with seed-derived masks (no mask storage) / GPU dropout без хранения масок
- CUDA Graph for decoder layers (off while dropout is active) / CUDA Graph для слоёв декодера (выкл. при dropout)
- Atomic checkpoint writes, v5 format with loss-scaler state / Атомарные чекпоинты v5

### Q: How do I run the GUI? / Как запустить GUI?
**A:** `.\scripts\windows\jgpt-gui.cmd`. Needs a JDK that bundles JavaFX (Liberica Full 25+); the script finds it under `~\.jdks`. Training is started as a child `jgpt-train-*.cmd`, so closing the GUI does not stop it. By default the script compiles only the `gui` package with javac into `target\gui-classes` (safe while a trainer is running); `--mvn` does a full Maven compile.  
**Ответ:** `jgpt-gui.cmd`; нужен JDK с JavaFX (Liberica Full). Обучение идёт дочерним процессом — закрытие GUI его не останавливает. По умолчанию собирается только пакет `gui` (безопасно при работающем тренере), `--mvn` — полная сборка.
