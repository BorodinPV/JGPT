#!/usr/bin/env bash
# Скачивает libcudnn (pip wheel) и headers cudnn-frontend в third_party/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CUDNN_DIR="$ROOT/third_party/cudnn"
FE_DIR="$ROOT/third_party/cudnn-frontend"

if [[ ! -f "$CUDNN_DIR/nvidia/cudnn/include/cudnn.h" ]]; then
  echo "[fetch-cudnn] pip install nvidia-cudnn-cu12 → $CUDNN_DIR"
  python3 -m pip install --target "$CUDNN_DIR" 'nvidia-cudnn-cu12'
fi

if [[ ! -f "$FE_DIR/include/cudnn_frontend.h" ]]; then
  echo "[fetch-cudnn] clone NVIDIA/cudnn-frontend v1.16.0"
  git clone --depth 1 --filter=blob:none --sparse --branch v1.16.0 \
    https://github.com/NVIDIA/cudnn-frontend.git "$FE_DIR"
  git -C "$FE_DIR" sparse-checkout set include
fi

echo "[fetch-cudnn] OK: $(ls "$CUDNN_DIR/nvidia/cudnn/lib/libcudnn.so"* 2>/dev/null | head -1)"
