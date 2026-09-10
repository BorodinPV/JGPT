# JGPT

GPT-модель (decoder-only transformer) с **полным обучением на GPU** через JNI + CUDA/cuBLAS.

> **Текущий прогон:** 37 слоёв, ~100M, SFT (JSONL). Каноническая геометрия без override — 12 слоёв, ~35M (книги).

---

## 🚀 Быстрый старт

**SFT ~100M (37 слоёв, vocab 16k, seq 2048)** — то, чем сейчас обычно учат:

```powershell
# Windows
.\scripts\windows\jgpt-train-37L-sft.ps1
```

```bash
# Linux
./scripts/linux/jgpt-train-37L-sft.sh
```

Данные: `.jsonl` в `data/sft/raw`. Resume: тот же скрипт (подхватит `checkpoint_final.bin`).

**Книги + авто-пресеты** (Linux, `LLMConfig.canonical()` ~35M):

```bash
./scripts/linux/jgpt-smart.sh
./scripts/linux/jgpt-smart.sh 02-stable
```

Положите `.txt` в `data/books/`. Карта скриптов: [scripts/README.md](scripts/README.md).

---

## 📐 Модель

| Параметр | Значение |
|----------|----------|
| Архитектура | Decoder-only GPT, pre-norm |
| Параметры | ~34.9M |
| d_model | 384 |
| Слои | 12 |
| Attention heads | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 1536) |
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

**RTX 3080 (10 GB VRAM)**

| Режим | Tokens/sec (порядок) |
|-------|----------------------|
| 12L, пресет `02-stable`, книги | ~26 000 |
| 37L SFT, seq 2048, batch 4×32 | ~9 000–10 000 |

Цифра 26k — старый 12-слойный прогон, не 37L.

---

## 📚 Документация

- [Архитектура и обучение](docs/training/README.md) — полное описание модели, пресетов, мониторинга
- [Тренировочный рунбук](docs/TRAIN_RUNBOOK.md) — практическое руководство по обучению
- [Поток данных 37L SFT](docs/data-flow-37L-sft.puml) — PlantUML (JSONL → GPU → чекпоинты)
- [FAQ](FAQ.md) — частые вопросы и решения проблем
- [Contributing](CONTRIBUTING.md) — как внести вклад в проект
- [Changelog](CHANGELOG.md) — история изменений

---

## 🛠 Системные требования

- **Java**: 25+ (с Vector API и preview-фичами). Maven должен запускаться на том же JDK (`JAVA_HOME`). На JDK 26 не используйте `--release 25` вместе с `--enable-preview`.
- **CUDA**: 12.x или 13.x с cuBLAS (`nvcc` ≠ готовая JNI-библиотека; её нужно собрать)
- **GPU**: FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+). Архитектура ядра — `native` (3080 → sm_86, 2060 → sm_75)
- **Linux**: CMake 3.24+, GCC ≤ 13 (или `-allow-unsupported-compiler`). Сборка: `./scripts/linux/build-cuda.sh` или `./scripts/linux/jgpt-smart.sh`
- **Windows**: CMake + Visual Studio 2022 Build Tools (MSVC/`cl.exe`) + CUDA Toolkit. Сборка: `.\scripts\windows\build-cuda.ps1`, затем `. .\build\jgpt-cuda-env.ps1`

---

## 📄 Лицензия

MIT — см. файл [LICENSE](LICENSE).

---

# JGPT

GPT model (decoder-only transformer) with **full GPU training** via JNI + CUDA/cuBLAS.

> **Current default run:** 37 layers, ~100M, SFT (JSONL). Geometry without env overrides is still 12 layers, ~35M (books).

---

## 🚀 Quick Start

**SFT ~100M (37 layers, vocab 16k, seq 2048):**

```powershell
# Windows
.\scripts\windows\jgpt-train-37L-sft.ps1
```

```bash
# Linux
./scripts/linux/jgpt-train-37L-sft.sh
```

Put `.jsonl` in `data/sft/raw`. Resume: run the same script (loads `checkpoint_final.bin`).

**Books + auto presets** (Linux, `LLMConfig.canonical()` ~35M):

```bash
./scripts/linux/jgpt-smart.sh
./scripts/linux/jgpt-smart.sh 02-stable
```

Place `.txt` files in `data/books/`. Script map: [scripts/README.md](scripts/README.md).

---

## 📐 Model

| Parameter | Value |
|-----------|-------|
| Architecture | Decoder-only GPT, pre-norm |
| Parameters | ~34.9M |
| d_model | 384 |
| Layers | 12 |
| Attention heads | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 1536) |
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

**RTX 3080 (10 GB VRAM)**

| Mode | Tokens/sec (ballpark) |
|------|----------------------|
| 12L, preset `02-stable`, books | ~26 000 |
| 37L SFT, seq 2048, batch 4×32 | ~9 000–10 000 |

The 26k figure is the old 12-layer run, not 37L SFT.

---

## 📚 Documentation

- [Architecture and Training](docs/training/README.md) — full description of the model, presets, monitoring
- [Training Runbook](docs/TRAIN_RUNBOOK.md) — practical guide to training
- [37L SFT data flow](docs/data-flow-37L-sft.puml) — PlantUML (JSONL → GPU → checkpoints)
- [FAQ](FAQ.md) — frequent questions and solutions
- [Contributing](CONTRIBUTING.md) — how to contribute to the project
- [Changelog](CHANGELOG.md) — changelog

---

## 🛠 System Requirements

- **Java**: 25+ (with Vector API and preview features). Maven must run on that same JDK (`JAVA_HOME`). On JDK 26 do not use `--release 25` together with `--enable-preview`.
- **CUDA**: 12.x or 13.x with cuBLAS (`nvcc` is not the JNI library; you must build it)
- **GPU**: FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+). Kernel arch is `native` (3080 → sm_86, 2060 → sm_75)
- **Linux**: CMake 3.24+, GCC ≤ 13 (or `-allow-unsupported-compiler`). Build: `./scripts/linux/build-cuda.sh` or `./scripts/linux/jgpt-smart.sh`
- **Windows**: CMake + Visual Studio 2022 Build Tools (MSVC/`cl.exe`) + CUDA Toolkit. Build: `.\scripts\windows\build-cuda.ps1`, then `. .\build\jgpt-cuda-env.ps1`

---

## 📄 License

MIT — see [LICENSE](LICENSE) file.