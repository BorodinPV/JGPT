# Changelog / История изменений

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added / Добавлено
- JavaFX desktop GUI (`scripts/windows/jgpt-gui.cmd`, package `com.veles.llm.jgpt.gui`): start/stop training, live loss/perplexity charts from `state/stats.json`, filtered log tail, checkpoint browser, in-process chat with any `model_*.bin`. Replaces the python-served `docs/dashboard.html` (kept, still works).
  - Десктопный GUI на JavaFX: обучение, графики, лог, чекпоинты, чат. Старый HTML-дашборд сохранён.
- GPU dropout on residual branches and embeddings (`JGPT_DROPOUT`, off by default): deterministic per-step/per-layer masks, no mask storage; decoder-layer CUDA Graph is disabled while active.
  - GPU dropout (residual + embedding) с детерминированными масками; CUDA Graph на слои выключается.
- Launcher flag `--restart-plan` (= `JGPT_FINETUNE=1` for one run): keep weights + Adam, reset step / LR schedule / epoch / best.
  - Флаг `--restart-plan`: веса и Adam остаются, план обучения с нуля.
- `JGPT_SAVE_EVERY_STEPS` / `JGPT_EVAL_EVERY_STEPS` (periodic `checkpoint_step_N`), `JGPT_DROPOUT`, `JGPT_FP16_AUX_SOFTEN_SAMPLE`.

### Changed / Изменено
- 28L-wide presets train with full-vocab CE (`JGPT_TRAIN_LOSS_MODE=full`): sampled CE with uniform negatives never pushed down plausible wrong tokens and was slower (~6k → ~12k tok/s on RTX 3080).
  - Wide-пресеты учатся полным CE: корректнее и вдвое быстрее sampled.
- Pretrain packs documents into one stream via `<eos>` (short docs and tails are no longer dropped) and splits train/val by document instead of by window.
  - Претрейн упаковывает документы через `<eos>`, val — по документам.
- Checkpoints: format v5 stores the FP16 loss-scaler state; all checkpoint/model/tokenizer writes are atomic (`.tmp` + rename); resume picks the newest of `final / step_N / epoch_N / best` by `globalStep` and loads the paired `model_*.bin`; pending writes are awaited on exit.
  - Чекпоинты v5, атомарная запись, resume по самому свежему шагу.
- AdamW weight decay applies only to rank ≥ 2 tensors (no decay on RMSNorm gains / 1-D params).
  - Weight decay только на матрицы.
- SFT preset no longer generates samples mid-run (`JGPT_INTERACTIVE_EVERY=0`): each sample divided the FP16 loss scale by 64 and drove it to 1.
  - В SFT выключены промежуточные сэмплы — они обваливали FP16 scale.
- Generation stops on `<user>`/`<assistant>` role tokens; sliding-window KV re-prefill uses position 0; BPE decode no longer inserts spaces before closing punctuation or inside hyphenated words.
  - Генерация останавливается на ролевых токенах; исправлены sliding KV и пробелы при декодировании.
- `jgpt-chat-28L-wide.cmd`: `--model <path>` and `--raw` are honoured (no forced chat template for the pretrain).
- Launch scripts are split by OS: `scripts/linux/*.sh`, `scripts/windows/*.ps1` (shared Python stays in `scripts/`). See `scripts/README.md`.
  - Скрипты запуска разделены: Linux `scripts/linux/`, Windows `scripts/windows/`.
- Native CUDA build is cross-platform: CMake uses `native` GPU arch, Windows builds one `jgpt_cuda.dll`, Linux still two `.so`. Scripts: `scripts/linux/build-cuda.sh`, `scripts/windows/build-cuda.ps1`.
  - Сборка CUDA и на Windows, и на Linux: arch `native`, на Windows одна DLL.
- Canonical model geometry (~34.9M): 12 layers, seq 1024, d_model 384, 24 heads, SwiGLU 1536. `LLMConfig.canonical()`; `smart50M()` is an alias. Preset `02-stable` no longer uses 20 layers.
  - Каноническая геометрия ~34.9M (12 слоёв, seq 1024, FFN 1536); пресет 02 больше не ставит 20 слоёв.
- Canonical GPU-train: CUDA implies resident weights, decoder pipeline, device logits and device decoder backward. Legacy `JGPT_TRAIN_GPU_RESIDENT` / `JGPT_FULL_GPU_TRAIN` / `JGPT_GPU_E2E_TRAIN` / `JGPT_DEVICE_LOGITS_TRAIN` / `JGPT_DEVICE_DECODER_BWD` / `JGPT_DECODER_GPU_PIPELINE` no longer select a path.
  - Канонический GPU-train: при CUDA полный путь без пяти env-флагов; старые `JGPT_*` GPU-переключатели игнорируются.
- Periodic VRAM cleanup/trim off by default (`JGPT_VRAM_CLEANUP_EVERY_STEPS=0`, `JGPT_CUDA_TRIM_EVERY_STEPS=0`). Eval/sample fences remain.
  - Периодический VRAM cleanup/trim выключены по умолчанию; барьеры после eval/sample сохранены.

### Added / Добавлено
- FlashAttention tile size configurable via `JGPT_FA_TILE_SIZE` environment variable
  - Размер плитки FlashAttention настраивается через переменную окружения `JGPT_FA_TILE_SIZE`
- Optimized Cross-Entropy kernel with block-per-row approach (12x faster)
  - Оптимизированное ядро Cross-Entropy с подходом block-per-row (в 12 раз быстрее)
- Warp-level reduction for embedding backward kernels
  - Редукция на уровне warp для embedding backward ядер
- CUDA Graph support for decoder layers
  - Поддержка CUDA Graph для слоёв декодера

### Changed / Изменено
- Improved training throughput from ~11k to ~26k tokens/s on RTX 3080
  - Повышена производительность обучения с ~11k до ~26k токенов/сек на RTX 3080
- Updated default FlashAttention tile size from 128 to 144 (reverted to 128 for RTX 3080 compatibility)
  - Обновлён размер плитки FlashAttention с 128 на 144 (возвращено на 128 для совместимости с RTX 3080)

### Fixed / Исправлено
- VRAM memory leak in LM head computation (cached buffer instead of malloc/free per step)
  - Утечка памяти VRAM в вычислении LM head (кэшированный буфер вместо malloc/free на каждом шаге)
- Embedding backward atomic contention reduced 32x with warp-level reduction
  - Конкуренция атомарных операций в embedding backward снижена в 32 раза с помощью warp-level редукции

---

## [1.0.0] - 2024-04-17

### Added / Добавлено
- Initial release of JGPT
  - Первоначальный релиз JGPT
- Full GPU training pipeline (forward, backward, optimizer on GPU)
  - Полный конвейер обучения на GPU (forward, backward, оптимизатор на GPU)
- FlashAttention-2 implementation
  - Реализация FlashAttention-2
- FP16 Tensor Cores support
  - Поддержка FP16 Tensor Cores
- cuBLAS integration for GEMM operations
  - Интеграция cuBLAS для GEMM операций
- 5 training presets (00-max-throughput to 04-minimal)
  - 5 пресетов обучения (от 00-max-throughput до 04-minimal)
- Auto-adaptive training script (jgpt-smart.sh)
  - Скрипт авто-адаптивного обучения (jgpt-smart.sh)
- BPE tokenizer
  - BPE токенизатор
- Checkpointing and resume functionality
  - Функциональность чекпоинтов и возобновления обучения
