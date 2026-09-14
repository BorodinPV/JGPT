# JGPT

GPT-модель (decoder-only transformer) с **полным обучением на GPU** через JNI + CUDA/cuBLAS.

> **Текущий прогон:** 28L-wide, ~134M — pretrain на ruwiki + классика (`data/books/pretrain_txt`), затем SFT на 60k коротких диалогов (`data/sft/short`). Полный CE по словарю, документы упакованы через `<eos>`, валидация по документам/диалогам, dropout 0.1. 37L SFT и 12L «книги» — старые пути, скрипты сохранены.

---

## 🚀 Быстрый старт

**28L-wide (~134M): pretrain → SFT → чат** (Windows):

```powershell
.\scripts\windows\jgpt-train-28L-wide.cmd --no-build        # pretrain, checkpoints\wide_28L_16k_1024
.\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build    # SFT от model_best.bin, checkpoints\wide_28L_sft
.\scripts\windows\jgpt-chat-28L-wide.cmd                    # чат с SFT model_best (или --raw для претрейна)
```

- **Остановка** — только `.\scripts\windows\jgpt-stop-train.cmd` (файл `state\STOP`, тренер допишет `checkpoint_final.bin`). Ctrl+C на Windows убивает java без чекпоинта.
- **Resume** — тот же скрипт без флагов: подхватит самый свежий из `checkpoint_final / step_N / epoch_N / best` по `globalStep`. `--fresh` — архив каталога и с нуля; `--restart-plan` — веса + Adam из чекпоинта, но шаг/LR/эпоха/best с нуля (после смены данных или пресета).
- **GUI** — `.\scripts\windows\jgpt-gui.cmd`: старт/стоп обучения, живые графики loss/perplexity, лог с фильтрами, каталог чекпоинтов и чат с любой моделью. JavaFX из Liberica Full JDK, ничего ставить не нужно.

Linux: `./scripts/linux/jgpt-train-28L-wide.sh`, `./scripts/linux/jgpt-train-28L-wide-sft.sh`.

**37L SFT ~100M** (JSONL из `data/sft/raw`, sampled CE — старый путь):

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```

**Книги + авто-пресеты** (Linux, `LLMConfig.canonical()` ~35M):

```bash
./scripts/linux/jgpt-smart.sh
./scripts/linux/jgpt-smart.sh 02-stable
```

Положите `.txt` в `data/books/`. Карта скриптов: [scripts/README.md](scripts/README.md).

---

## 📐 Модель

| Параметр | 28L-wide (текущая) | canonical (без override) |
|----------|--------------------|--------------------------|
| Архитектура | Decoder-only GPT, pre-norm | то же |
| Параметры | ~134M | ~34.9M |
| d_model | 512 | 384 |
| Слои | 28 | 12 |
| Attention heads | 32 (d_head = 16) | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 2048) | SwiGLU (1536) |
| Контекст | 1024 токена | 1024 |
| Токенизация | BPE 16k, без lowercasing, `<user>`/`<assistant>` | BPE 8k |
| Позиции | RoPE | RoPE |
| Нормализация | RMSNorm | RMSNorm |
| Loss | полный CE по словарю (`JGPT_TRAIN_LOSS_MODE=full`) | по пресету |
| Регуляризация | dropout 0.1 (residual + embedding), AdamW weight decay только на матрицы | — |

Пресеты геометрии — `env/*.env`; `JGPT_MAX_SEQ_LEN` / `JGPT_PRESET_NUM_LAYERS` / `JGPT_D_MODEL` должны совпадать с чекпоинтом.

---

## ⚡ GPU-ускорение

- **FP16 GEMM** — cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** — cuDNN SDPA (fused, FP16), fallback WMMA
- **Полный CE по словарю на GPU** — логиты и ∂CE на устройстве, один GEMM на LM head (быстрее sampled-пути и без смещения objective)
- **Fused-операции** — RMSNorm + FFN, RMSNorm + LM head via cuBLAS
- **Полный GPU-цикл** — forward, backward, optimiser — всё на VRAM (канонический путь при CUDA; отдельные `JGPT_*` GPU-флаги больше не нужны)
- **Decoder pipeline** — слой-за-слоем без D2H
- **GPU dropout** — детерминированные маски по seed шага/слоя, без хранения масок; при dropout CUDA Graph на слои отключается
- **Атомарные чекпоинты** — запись во `.tmp` + rename; формат v5 хранит Adam, позицию в эпохе и состояние FP16 loss-scaler

---

## 📊 Производительность

**RTX 3080 (10 GB VRAM)**

| Режим | Tokens/sec (порядок) |
|-------|----------------------|
| 28L-wide, seq 1024, batch 4×16, полный CE + dropout | ~12 000 (шаг ≈ 5.3 с) |
| 28L-wide SFT, batch 4×8 | ~12 000 (шаг ≈ 2.7 с) |
| 37L SFT, seq 2048, batch 4×32, sampled CE | ~9 000–10 000 |
| 12L, пресет `02-stable`, книги | ~26 000 |

Тот же 28L-wide на sampled CE (384 кандидата) давал ~6 000 — gather-голова без Tensor Cores медленнее одного полного GEMM.

---

## 📚 Документация

- [Архитектура и обучение](docs/training/README.md) — пресеты, resume, dropout, мониторинг
- [Тренировочный рунбук](docs/TRAIN_RUNBOOK.md) — практическое руководство по обучению
- [Подготовка данных](docs/DATA_PREPARATION.md) — очистка текстов
- [Поток данных 37L SFT](docs/data-flow-37L-sft.puml) — PlantUML (JSONL → GPU → чекпоинты)
- [FAQ](FAQ.md) — частые вопросы и решения проблем
- [Contributing](CONTRIBUTING.md) — как внести вклад в проект
- [Changelog](CHANGELOG.md) — история изменений

Мониторинг: `scripts\windows\jgpt-gui.cmd` (JavaFX). Старый `docs/dashboard.html` читает тот же `state/stats.json`, но требует `python -m http.server`.

---

## 🛠 Системные требования

- **Java**: 25+ (с Vector API и preview-фичами). Maven должен запускаться на том же JDK (`JAVA_HOME`). На JDK 26 не используйте `--release 25` вместе с `--enable-preview`. Для GUI нужен JDK с JavaFX (Liberica **Full**); без него обучение и чат из консоли работают как прежде.
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

> **Current run:** 28L-wide, ~134M — pretrain on ruwiki + classics (`data/books/pretrain_txt`), then SFT on 60k short dialogs (`data/sft/short`). Full-vocab CE, documents packed via `<eos>`, validation split by document/dialog, dropout 0.1. 37L SFT and the 12L "books" path are legacy; their scripts remain.

---

## 🚀 Quick Start

**28L-wide (~134M): pretrain → SFT → chat** (Windows):

```powershell
.\scripts\windows\jgpt-train-28L-wide.cmd --no-build        # pretrain, checkpoints\wide_28L_16k_1024
.\scripts\windows\jgpt-train-28L-wide-sft.cmd --no-build    # SFT seeded from model_best.bin, checkpoints\wide_28L_sft
.\scripts\windows\jgpt-chat-28L-wide.cmd                    # chat with SFT model_best (or --raw for the pretrain)
```

- **Stop** only via `.\scripts\windows\jgpt-stop-train.cmd` (creates `state\STOP`; the trainer writes `checkpoint_final.bin` and exits). Ctrl+C on Windows kills java without a checkpoint.
- **Resume** — same script, no flags: picks the newest of `checkpoint_final / step_N / epoch_N / best` by `globalStep`. `--fresh` archives the checkpoint dir and starts over; `--restart-plan` keeps weights + Adam but resets step/LR/epoch/best (after changing data or preset).
- **GUI** — `.\scripts\windows\jgpt-gui.cmd`: start/stop training, live loss/perplexity charts, filtered log tail, checkpoint browser and chat with any model. JavaFX ships with Liberica Full JDK, nothing to install.

Linux: `./scripts/linux/jgpt-train-28L-wide.sh`, `./scripts/linux/jgpt-train-28L-wide-sft.sh`.

**37L SFT ~100M** (JSONL from `data/sft/raw`, sampled CE — legacy path):

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```

**Books + auto presets** (Linux, `LLMConfig.canonical()` ~35M):

```bash
./scripts/linux/jgpt-smart.sh
./scripts/linux/jgpt-smart.sh 02-stable
```

Place `.txt` files in `data/books/`. Script map: [scripts/README.md](scripts/README.md).

---

## 📐 Model

| Parameter | 28L-wide (current) | canonical (no overrides) |
|-----------|--------------------|--------------------------|
| Architecture | Decoder-only GPT, pre-norm | same |
| Parameters | ~134M | ~34.9M |
| d_model | 512 | 384 |
| Layers | 28 | 12 |
| Attention heads | 32 (d_head = 16) | 24 (d_head = 16) |
| FFN | SwiGLU (d_intermediate = 2048) | SwiGLU (1536) |
| Context | 1024 tokens | 1024 |
| Tokenization | BPE 16k, no lowercasing, `<user>`/`<assistant>` | BPE 8k |
| Positions | RoPE | RoPE |
| Normalization | RMSNorm | RMSNorm |
| Loss | full-vocab CE (`JGPT_TRAIN_LOSS_MODE=full`) | per preset |
| Regularization | dropout 0.1 (residual + embedding), AdamW weight decay on matrices only | — |

Geometry presets live in `env/*.env`; `JGPT_MAX_SEQ_LEN` / `JGPT_PRESET_NUM_LAYERS` / `JGPT_D_MODEL` must match the checkpoint.

---

## ⚡ GPU Acceleration

- **FP16 GEMM** — cuBLAS GemmEx + Tensor Cores
- **FlashAttention-2** — cuDNN SDPA (fused, FP16), WMMA fallback
- **Full-vocab CE on GPU** — logits and ∂CE on device, one GEMM for the LM head (faster than the sampled path and an unbiased objective)
- **Fused operations** — RMSNorm + FFN, RMSNorm + LM head via cuBLAS
- **Full GPU cycle** — forward, backward, optimiser — all in VRAM (canonical path when CUDA is available; separate `JGPT_*` GPU flags are no longer required)
- **Decoder pipeline** — layer-by-layer without D2H
- **GPU dropout** — deterministic masks from a per-step/per-layer seed, no mask storage; decoder-layer CUDA Graph is disabled while dropout is active
- **Atomic checkpoints** — write to `.tmp` + rename; format v5 carries Adam, epoch position and the FP16 loss-scaler state

---

## 📊 Performance

**RTX 3080 (10 GB VRAM)**

| Mode | Tokens/sec (ballpark) |
|------|----------------------|
| 28L-wide, seq 1024, batch 4×16, full CE + dropout | ~12 000 (step ≈ 5.3 s) |
| 28L-wide SFT, batch 4×8 | ~12 000 (step ≈ 2.7 s) |
| 37L SFT, seq 2048, batch 4×32, sampled CE | ~9 000–10 000 |
| 12L, preset `02-stable`, books | ~26 000 |

The same 28L-wide with sampled CE (384 candidates) ran at ~6 000: the gather LM head without Tensor Cores is slower than one full GEMM.

---

## 📚 Documentation

- [Architecture and Training](docs/training/README.md) — presets, resume, dropout, monitoring
- [Training Runbook](docs/TRAIN_RUNBOOK.md) — practical guide to training
- [Data preparation](docs/DATA_PREPARATION.md) — text cleanup
- [37L SFT data flow](docs/data-flow-37L-sft.puml) — PlantUML (JSONL → GPU → checkpoints)
- [FAQ](FAQ.md) — frequent questions and solutions
- [Contributing](CONTRIBUTING.md) — how to contribute to the project
- [Changelog](CHANGELOG.md) — changelog

Monitoring: `scripts\windows\jgpt-gui.cmd` (JavaFX). The legacy `docs/dashboard.html` reads the same `state/stats.json` but needs `python -m http.server`.

---

## 🛠 System Requirements

- **Java**: 25+ (with Vector API and preview features). Maven must run on that same JDK (`JAVA_HOME`). On JDK 26 do not use `--release 25` together with `--enable-preview`. The GUI needs a JDK bundling JavaFX (Liberica **Full**); training and console chat work without it.
- **CUDA**: 12.x or 13.x with cuBLAS (`nvcc` is not the JNI library; you must build it)
- **GPU**: FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+). Kernel arch is `native` (3080 → sm_86, 2060 → sm_75)
- **Linux**: CMake 3.24+, GCC ≤ 13 (or `-allow-unsupported-compiler`). Build: `./scripts/linux/build-cuda.sh` or `./scripts/linux/jgpt-smart.sh`
- **Windows**: CMake + Visual Studio 2022 Build Tools (MSVC/`cl.exe`) + CUDA Toolkit. Build: `.\scripts\windows\build-cuda.ps1`, then `. .\build\jgpt-cuda-env.ps1`

---

## 📄 License

MIT — see [LICENSE](LICENSE) file.