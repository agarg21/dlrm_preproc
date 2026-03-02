#!/usr/bin/env bash
# scripts/run_training.sh
# End-to-end: export preproc → train DLRM
set -euo pipefail

PYTHON=${PYTHON:-/opt/homebrew/bin/python3.11}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Step 1: Export TorchScript preproc module ==="
$PYTHON -m preproc.export --out artifacts/preproc.pt

echo ""
echo "=== Step 2: Train DLRM ==="
$PYTHON -m trainer.train \
    --epochs 3 \
    --steps  200 \
    --batch  128 \
    --emb-dim 64 \
    --lr 0.01 \
    --log-every 50 \
    --preproc artifacts/preproc.pt

echo ""
echo "=== Done ==="
