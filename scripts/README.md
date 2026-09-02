# JGPT scripts

Launchers are split by OS. Shared helpers (Python) stay in this directory.

| Path | OS | Role |
|------|----|------|
| `linux/jgpt-train-37L-sft.sh` | Linux | 37L ~100M SFT (JSONL, resume from checkpoint) |
| `windows/jgpt-train-37L-sft.ps1` | Windows | Same preset (`windows/jgpt-train-37L-sft.cmd` wrapper) |
| `linux/jgpt-smart.sh` | Linux | Books corpus, auto-switch `env/00`…`04` |
| `linux/jgpt-train-24L.sh` / `32L.sh` | Linux | Fixed book presets |
| `linux/jgpt-train-32L-sft.sh` | Linux | Alias → `jgpt-train-37L-sft.sh` |
| `linux/build-cuda.sh` | Linux | JNI CUDA `.so` |
| `windows/build-cuda.ps1` | Windows | JNI CUDA `.dll` |
| `linux/fetch-cudnn.sh` | Linux | pip `libcudnn.so.9` |
| `windows/fetch-cudnn.ps1` | Windows | win_amd64 wheel + `cudnn.lib` |
| `linux/jgpt-chat.sh` | Linux | Interactive chat after train |
| `sft-export-jsonl.py` | both | Optional parquet → JSONL |
| `clean-libru-txt.py` | both | lib.ru text cleanup |

From the repo root:

```bash
./scripts/linux/jgpt-train-37L-sft.sh
```

```powershell
.\scripts\windows\jgpt-train-37L-sft.ps1
```
