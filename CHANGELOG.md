# Changelog / История изменений

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

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
