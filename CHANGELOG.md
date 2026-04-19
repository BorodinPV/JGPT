# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- FlashAttention tile size configurable via `JGPT_FA_TILE_SIZE` environment variable
- Optimized Cross-Entropy kernel with block-per-row approach (12x faster)
- Warp-level reduction for embedding backward kernels
- CUDA Graph support for decoder layers

### Changed
- Improved training throughput from ~11k to ~26k tokens/s on RTX 3080
- Updated default FlashAttention tile size from 128 to 144 (reverted to 128 for RTX 3080 compatibility)

### Fixed
- VRAM memory leak in LM head computation (cached buffer instead of malloc/free per step)
- Embedding backward atomic contention reduced 32x with warp-level reduction

## [1.0.0] - 2024-04-17

### Added
- Initial release of JGPT
- Full GPU training pipeline (forward, backward, optimizer on GPU)
- FlashAttention-2 implementation
- FP16 Tensor Cores support
- cuBLAS integration for GEMM operations
- 5 training presets (00-max-throughput to 04-minimal)
- Auto-adaptive training script (jgpt-smart.sh)
- BPE tokenizer
- Checkpointing and resume functionality
