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
- **FlashAttention-2** — fused QKV attention
- **Fused-операции** — RMSNorm + FFN, RMSNorm + LM head
- **Полный GPU-цикл** — forward, backward, optimiser — всё на VRAM
- **Decoder pipeline** — слой-за-слоем без D2H
- **CUDA Graph** — на слои декодера (опционально)
- **Async checkpointing** — веса пишутся в фоне

---

## 📊 Производительность

**RTX 3080 (10 GB VRAM)**, пресет 02-stable:

| Метрика | Значение |
|---------|----------|
| Tokens/sec | ~11 500 |
| Шаг | ~715 мс (forward 368 + CE 4 + backward 314 + optimiser 29) |
| VRAM | ~3700 / 9873 МБ |

---

## 📚 Документация

- [Архитектура и обучение](docs/README.md) — полное описание модели, пресетов, мониторинга
- [Переменные окружения](docs/ENVIRONMENT.md) — справочник всех `JGPT_*` параметров
- [Тренировочный рунбук](docs/TRAIN_RUNBOOK.md) — практическое руководство по обучению

---

## 🛠 Системные требования

- **Java**: 25+ (с Vector API и preview-фичами)
- **CUDA**: 12.x с cuBLAS
- **GPU**: с поддержкой FP16 Tensor Cores (RTX 20xx+, RTX 30xx+, A100+)
- **Контроллер**: GCC ≤ 13 (или `-allow-unsupported-compiler`)

---

## 📄 Лицензия

MIT — см. файл [LICENSE](LICENSE).
