#!/usr/bin/env bash
# scripts/build_cpp.sh
# Build the C++ preproc_runner binary using LibTorch from the active Python env.
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve LibTorch cmake prefix from the active Python env
TORCH_CMAKE=$($PYTHON -c "import torch; print(torch.utils.cmake_prefix_path)")
echo "Using LibTorch at: $TORCH_CMAKE"

BUILD_DIR="$ROOT/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build . --config Release -j"$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"

echo ""
echo "Build complete.  Binary: $BUILD_DIR/preproc_runner"
echo "Run:  ./build/preproc_runner artifacts/preproc.pt"
