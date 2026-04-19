# Contributing to JGPT

Thank you for your interest in contributing to JGPT!

## Development Setup

### Requirements
- Java 25+ with Vector API
- CUDA 12.x with cuBLAS
- GCC ≤ 13 (or use `-allow-unsupported-compiler` flag for GCC 14+)
- Maven 3.9+

### Build
```bash
# Clone repository
git clone <repository-url>
cd JGPT

# Build native libraries
cd src/main/cpp
mkdir -p build && cd build
cmake ..
cmake --build .
cd ../../..

# Build Java code
mvn compile

# Run tests
mvn test
```

### Build with custom FlashAttention tile size
```bash
# For GPUs with limited shared memory (RTX 3080)
cd build
JGPT_FA_TILE_SIZE=128 cmake ..
cmake --build .
```

## Code Style

### Java
- Follow standard Java conventions
- Use `final` where possible
- Document public APIs with Javadoc

### C++/CUDA
- Use `constexpr` for compile-time constants
- Prefer `__restrict__` for pointer parameters
- Document kernels with block/thread dimensions

## Testing

Before submitting PR:
1. Run full training for at least 100 steps
2. Verify no CUDA errors in logs
3. Check performance metrics (should be ~25k+ tokens/s on RTX 3080)

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Reporting Issues

When reporting bugs, please include:
- GPU model and CUDA version
- Java version (`java -version`)
- GCC version (`gcc --version`)
- Error messages from `training_allbooks.log`
- Steps to reproduce

## Questions?

Join discussions in GitHub Issues or check the documentation in `docs/`.
