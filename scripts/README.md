# JGPT scripts

Launchers are split by OS. Shared helpers (Python) stay in this directory.

| Path | OS | Role |
|------|----|------|
| `linux/jgpt-train-28L-wide.sh` | Linux | Wide 28L books pretrain (`env/28L-wide-pretrain.env`) |
| `linux/jgpt-train-28L-wide-sft.sh` | Linux | SFT after 28L-wide (`env/28L-wide-sft.env`) |
| `windows/jgpt-train-28L-wide.ps1` | Windows | Same pretrain (`windows/jgpt-train-28L-wide.cmd`) |
| `windows/jgpt-train-28L-wide-sft.ps1` | Windows | Same SFT (`windows/jgpt-train-28L-wide-sft.cmd`) |
| `windows/jgpt-chat-28L-wide.ps1` | Windows | InferChat on 28L-wide SFT/pretrain |
| `linux/jgpt-train-20L-wide.sh` | Linux | Wide 20L books pretrain (`env/20L-wide-pretrain.env`) |
| `linux/jgpt-train-20L-wide-sft.sh` | Linux | SFT after 20L-wide (`env/20L-wide-sft.env`) |
| `windows/jgpt-train-20L-wide.ps1` | Windows | Same pretrain (`windows/jgpt-train-20L-wide.cmd`) |
| `windows/jgpt-train-20L-wide-sft.ps1` | Windows | Same SFT (`windows/jgpt-train-20L-wide-sft.cmd`) |
| `windows/jgpt-chat-20L-wide.ps1` | Windows | InferChat on 20L-wide SFT/pretrain |
| `linux/jgpt-train-37L-sft.sh` | Linux | 37L ~100M SFT (JSONL, resume from checkpoint) |
| `linux/jgpt-train-37L-sft-short.sh` | Linux | Short Q&A finetune from `model_best.bin` (`env/37L-sft-short-ft.env`) |
| `linux/jgpt-train-37L-sft-exam.sh` | Linux | Tiny clean exam finetune (`env/37L-sft-exam.env`) |
| `windows/jgpt-train-37L-sft.ps1` | Windows | Same preset (`windows/jgpt-train-37L-sft.cmd` wrapper) |
| `windows/jgpt-train-37L-sft-short.ps1` | Windows | Short Q&A finetune (`windows/jgpt-train-37L-sft-short.cmd`) |
| `windows/jgpt-train-37L-sft-exam.ps1` | Windows | Tiny clean exam finetune (`windows/jgpt-train-37L-sft-exam.cmd`) |
| `windows/jgpt-chat-37L-sft.ps1` | Windows | InferChat on `model_best.bin` (`windows/jgpt-chat-37L-sft.cmd`) |
| `windows/jgpt-chat-37L-sft-short.ps1` | Windows | InferChat on short-ft `model_best.bin` |
| `windows/jgpt-chat-37L-sft-exam.ps1` | Windows | InferChat on exam-ft `model_best.bin` |
| `linux/jgpt-smart.sh` | Linux | Books corpus, auto-switch `env/00`…`04` |
| `linux/jgpt-train-24L.sh` / `32L.sh` | Linux | Fixed book presets |
| `linux/jgpt-train-32L-sft.sh` | Linux | Alias → `jgpt-train-37L-sft.sh` |
| `linux/build-cuda.sh` | Linux | JNI CUDA `.so` |
| `windows/build-cuda.ps1` | Windows | JNI CUDA `.dll` |
| `linux/fetch-cudnn.sh` | Linux | pip `libcudnn.so.9` |
| `windows/fetch-cudnn.ps1` | Windows | win_amd64 wheel + `cudnn.lib` |
| `linux/jgpt-chat.sh` | Linux | Interactive chat after train |
| `sft-export-jsonl.py` | both | Optional parquet → JSONL |
| `sft-filter-short.py` | both | Filter JSONL to short assistant replies → `data/sft/short` |
| `sft-make-exam.py` | both | Tiny clean exam JSONL → `data/sft/exam` |
| `fetch-ru-pretrain.py` | both | Starter corpus (ruwiki dump by default) → `data/books/pretrain_txt` |
| `clean-libru-txt.py` | both | lib.ru text cleanup |

From the repo root:

```bash
./scripts/linux/jgpt-train-37L-sft.sh
```

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```
