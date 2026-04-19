# Frequently Asked Questions

## Performance

### Q: What throughput should I expect?
**A:** On RTX 3080 (10GB) with preset 02-stable:
- ~26,000 tokens/sec
- ~1250ms per step (forward 600ms + backward 620ms + optimizer 30ms)

### Q: How does JGPT compare to PyTorch?
**A:** JGPT achieves ~1.0-1.2x PyTorch performance for similar models on same hardware, due to:
- Custom optimized CUDA kernels
- No Python overhead
- Direct cuBLAS integration

## Build Issues

### Q: GCC 15 is not supported by CUDA
**A:** Add `-allow-unsupported-compiler` flag to CMakeLists.txt or use GCC ≤ 13.

### Q: `cudaFuncSetAttribute smem=110592: invalid argument`
**A:** FlashAttention tile size is too large for your GPU. Use smaller tile:
```bash
JGPT_FA_TILE_SIZE=128 cmake ..
```

**Maximum tile sizes by GPU:**
- RTX 3080/4090: 128 (82KB shared memory)
- A100: 144-160 (110-136KB)
- H100: 192+ (184KB+)

## Training

### Q: Out of memory error
**A:** Try in order:
1. Switch to preset 03-recovery or 04-minimal
2. Reduce `JGPT_BATCH_SIZE` to 1
3. Reduce `JGPT_MAX_SEQ_LEN` to 512
4. Disable CUDA Graph: `JGPT_DECODER_LAYER_CUDA_GRAPH=0`

### Q: Training stopped with "overflow-скип"
**A:** FP16 scale is stuck. Training will auto-downgrade preset. You can also:
- Reduce `JGPT_FP16_DYNAMIC_INITIAL`
- Increase `JGPT_FP16_DYNAMIC_GROWTH_INTERVAL`

### Q: How to resume training?
**A:** Just run `./scripts/jgpt-smart.sh` again. It will auto-resume from `checkpoint_final.bin`.

## Configuration

### Q: What presets are available?
| Preset | Use case | Batch | CUDA Graph |
|--------|----------|-------|------------|
| 00-max-throughput | Maximum speed | 4 | Off (OOM risk) |
| 01-aggressive | Default start | 1 | On |
| 02-stable | Stable training | 2 | On |
| 03-recovery | After OOM | 1 | Off |
| 04-minimal | Last resort | 1 | Off |

### Q: How to change FlashAttention tile size?
**A:** Tile size is compile-time constant. Rebuild with:
```bash
cd build
JGPT_FA_TILE_SIZE=128 cmake ../src/main/cpp
cmake --build .
```

Valid values: 64, 96, 128, 144 (A100+), 160 (A100+), 192 (H100+)

## Architecture

### Q: Why is backward pass slower than forward?
**A:** Normal for transformers. Backward requires:
- Gradient computation for all parameters
- Weight updates
- More memory bandwidth

Typical ratio: backward = 1.0-1.2x forward time.

### Q: What optimizations are implemented?
- FlashAttention-2 (fused attention)
- Block-per-row Cross-Entropy (12x faster)
- Warp-level reduction for embeddings
- FP16 Tensor Cores for GEMM
- CUDA Graph for decoder layers
- Async checkpointing
