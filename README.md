# JGPT

GPT-модель (decoder-only transformer) с **полным обучением на GPU** через JNI + CUDA/cuBLAS.

> **~35M параметров** · **FlashAttention-2** · **FP16 Tensor Cores** · **CUDA Graph**

---

## 🚀 Быстрый старт

```bash
# Сборка + обучение (одна команда)
./scripts/jgpt-smart.sh

# С конкретным пресетом
./scripts/jgpt-smart.sh 02-stable
```

Положите `.txt` файлы в `data/books/` — и обучение начнёт с них.

---

## 📐 Модель

| Параметр | Значение |
|----------|----------|
| Архитектура | Decoder-only GPT, pre-norm |
| Параметры | ~34.9M |
| d_model | 384 |
| Слои | 12 |
| Attention heads | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 768) |
| Контекст | 1024 токена |
| Токенизация | BPE, vocab = 8000 |
| Позиции | RoPE |
| Нормализация | RMSNorm |

---

## ⚡ GPU-ускорение

- **FP16 GEMM** — cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** — fused QKV attention (tile size 128)
- **Optimized kernels** — block-per-row CE, warp-level reduction for embeddings
- **Fused-операции** — RMSNorm + FFN, RMSNorm + LM head via cuBLAS
- **Полный GPU-цикл** — forward, backward, optimiser — всё на VRAM (канонический путь при CUDA; отдельные `JGPT_*` GPU-флаги больше не нужны)
- **Decoder pipeline** — слой-за-слоем без D2H
- **CUDA Graph** — на слои декодера (опционально)
- **Async checkpointing** — веса пишутся в фоне

---

## 📊 Производительность

**RTX 3080 (10 GB VRAM)**, пресет 02-stable:

| Метрика | Значение |
|---------|----------|
| Tokens/sec | ~26 000 |
| Шаг | ~1250 мс (forward 600 + CE 9 + backward 620 + optimiser 29) |
| VRAM | ~5200 / 10000 МБ |

---

## 📚 Документация

- [Архитектура и обучение](docs/training/README.md) — полное описание модели, пресетов, мониторинга
- [Тренировочный рунбук](docs/TRAIN_RUNBOOK.md) — практическое руководство по обучению
- [FAQ](FAQ.md) — частые вопросы и решения проблем
- [Contributing](CONTRIBUTING.md) — как внести вклад в проект
- [Changelog](CHANGELOG.md) — история изменений

---

## 🛠 Системные требования

- **Java**: 25+ (с Vector API и preview-фичами). Maven должен запускаться на том же JDK (`JAVA_HOME`). На JDK 26 не используйте `--release 25` вместе с `--enable-preview`.
- **CUDA**: 12.x или 13.x с cuBLAS (`nvcc` ≠ готовая JNI-библиотека; её нужно собрать)
- **GPU**: FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+). Архитектура ядра — `native` (3080 → sm_86, 2060 → sm_75)
- **Linux**: CMake 3.24+, GCC ≤ 13 (или `-allow-unsupported-compiler`). Сборка: `./scripts/build-cuda.sh` или `./scripts/jgpt-smart.sh`
- **Windows**: CMake + Visual Studio 2022 Build Tools (MSVC/`cl.exe`) + CUDA Toolkit. Сборка: `.\scripts\build-cuda.ps1`, затем `. .\build\jgpt-cuda-env.ps1`

---

## 📄 Лицензия

MIT — см. файл [LICENSE](LICENSE).

---

# JGPT

GPT model (decoder-only transformer) with **full GPU training** via JNI + CUDA/cuBLAS.

> **~35M parameters** · **FlashAttention-2** · **FP16 Tensor Cores** · **CUDA Graph**

---

## 🚀 Quick Start

```bash
# Build + train (one command)
./scripts/jgpt-smart.sh

# With a specific preset
./scripts/jgpt-smart.sh 02-stable
```

Place `.txt` files in `data/books/` — training will start on them.

---

## 📐 Model

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only GPT, pre-norm |
| Parameters | ~34.9M |
| d_model | 384 |
| Layers | 12 |
| Attention heads | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 768) |
| Context | 1024 tokens |
| Tokenization | BPE, vocab = 8000 |
| Positions | RoPE |
| Normalization | RMSNorm |

---

## ⚡ GPU Acceleration

- **FP16 GEMM** — cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** — fused QKV attention (tile size 128)
- **Optimized kernels** — block-per-row CE, warp-level reduction for embeddings
- **Fused operations** — RMSNorm + FFN, RMSNorm + LM head via cuBLAS
- **Full GPU cycle** — forward, backward, optimiser — all in VRAM (canonical path when CUDA is available; separate `JGPT_*` GPU flags are no longer required)
- **Decoder pipeline** — layer-by-layer without D2H
- **CUDA Graph** — on decoder layers (optional)
- **Async checkpointing** — weights written in background

---

## 📊 Performance

**RTX 3080 (10 GB VRAM)**, preset 02-stable:

| Metric | Value |
|--------|-------|
| Tokens/sec | ~26 000 |
| Step | ~1250 ms (forward 600 + CE 9 + backward 620 + optimiser 29) |
| VRAM | ~5200 / 10000 MB |

---

## 📚 Documentation

- [Architecture and Training](docs/training/README.md) — full description of the model, presets, monitoring
- [Training Runbook](docs/TRAIN_RUNBOOK.md) — practical guide to training
- [FAQ](FAQ.md) — frequent questions and solutions
- [Contributing](CONTRIBUTING.md) — how to contribute to the project
- [Changelog](CHANGELOG.md) — changelog

---

## 🛠 System Requirements

- **Java**: 25+ (with Vector API and preview features). Maven must run on that same JDK (`JAVA_HOME`). On JDK 26 do not use `--release 25` together with `--enable-preview`.
- **CUDA**: 12.x or 13.x with cuBLAS (`nvcc` is not the JNI library; you must build it)
- **GPU**: FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+). Kernel arch is `native` (3080 → sm_86, 2060 → sm_75)
- **Linux**: CMake 3.24+, GCC ≤ 13 (or `-allow-unsupported-compiler`). Build: `./scripts/build-cuda.sh` or `./scripts/jgpt-smart.sh`
- **Windows**: CMake + Visual Studio 2022 Build Tools (MSVC/`cl.exe`) + CUDA Toolkit. Build: `.\scripts\build-cuda.ps1`, then `. .\build\jgpt-cuda-env.ps1`

---

## 📄 License

MIT — see [LICENSE](LICENSE) file.