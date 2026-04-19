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
- **Полный GPU-цикл** — forward, backward, optimiser — всё на VRAM
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

- **Java**: 25+ (с Vector API и preview-фичами)
- **CUDA**: 12.x с cuBLAS
- **GPU**: с поддержкой FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+)
- **Контроллер**: GCC ≤ 13 (или `-allow-unsupported-compiler`)

---

## 📄 Лицензия

MIT — см. файл [LICENSE](LICENSE).
