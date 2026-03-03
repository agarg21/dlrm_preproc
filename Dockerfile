# Dockerfile — dlrm_preproc
#
# Multi-stage image:
#   Stage 1 (base)    — Python + PyTorch CPU + project deps
#   Stage 2 (cpp)     — adds cmake / g++ for the C++ preproc_runner binary
#   Stage 3 (runtime) — minimal image that ships both the Python harness
#                       and the pre-built C++ binary
#
# Build targets
#   docker build --target runtime -t dlrm_preproc .          # default: python only
#   docker build --target cpp     -t dlrm_preproc:cpp .      # includes C++ binary
#
# Usage examples are in CLAUDE.md.
# ---------------------------------------------------------------------------

# ── Stage 1: Python + PyTorch (CPU) ────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by PyTorch CPU wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel first (≈800 MB) so layer is cached
# independently of the rest of the source code.
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Copy project source (after heavy deps so changes don't bust PyTorch layer)
COPY preproc/    preproc/
COPY data/       data/
COPY model/      model/
COPY trainer/    trainer/
COPY requirements.txt .

# requirements.txt only pins torch>=2.0 (already installed above)
RUN pip install --no-cache-dir -r requirements.txt

# Create artifacts dir; the .pt file is volume-mounted or generated at runtime
RUN mkdir -p artifacts && touch artifacts/.gitkeep

# ── Stage 2: C++ build env ──────────────────────────────────────────────────
FROM base AS cpp

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY preproc_runner/ preproc_runner/
COPY CMakeLists.txt  .

# Build the preproc_runner binary using LibTorch bundled with the pip wheel
RUN CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
    && cmake -B build \
             -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
             -G Ninja \
    && cmake --build build --target preproc_runner -- -j$(nproc)

# ── Stage 3: Runtime (default target) ──────────────────────────────────────
# Copies just the Python source + optionally the C++ binary.
# Default target is Python-only; use --target cpp if you need the binary.
FROM base AS runtime

# Copy the compiled C++ binary from the cpp stage (optional convenience)
COPY --from=cpp /app/build/preproc_runner /app/build/preproc_runner

# Add the LibTorch shared libs so the binary can run without the full dev env
COPY --from=cpp /usr/local/lib/python3.11/site-packages/torch/lib/ \
                /usr/local/lib/python3.11/site-packages/torch/lib/

# Default command: export preproc module then run training
CMD ["python", "-m", "trainer.train", "--epochs", "3", "--steps", "200", "--batch", "128"]
