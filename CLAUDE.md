# CLAUDE.md — dlrm_preproc

Context file for Claude Code. Read this before editing any file in this repo.

## Project purpose

Training harness for a DLRM-like recommendation model with feature preprocessing
running as a TorchScript module — loadable in both Python (training) and C++ (serving).

Core design: the **same** `artifacts/preproc.pt` file is used by:
1. `PreprocService` in Python training (zero-copy, in-process)
2. `preproc_runner` C++ binary (standalone, production serving path)

## Repo layout

```
dlrm_preproc/
├── preproc/
│   ├── module.py       # DensePreproc, SparsePreproc, FeaturePreproc (scriptable)
│   └── export.py       # scripts + saves to artifacts/preproc.pt
├── preproc_runner/
│   └── main.cpp        # C++ binary: load .pt, run on synthetic batch
├── data/
│   └── synthetic.py    # CriteoSynthetic IterableDataset + collate_fn
├── model/
│   └── dlrm.py         # TorchRec-mirrored DLRM (EmbeddingBagCollection, DLRMTrain)
├── trainer/
│   └── train.py        # Full training loop (data → preproc → model → backward)
├── scripts/
│   ├── build_cpp.sh    # Builds preproc_runner via CMake + LibTorch
│   └── run_training.sh # End-to-end: export preproc → train
├── CMakeLists.txt      # C++ build config
├── artifacts/          # Generated files (gitignored except .gitkeep)
└── requirements.txt
```

## Key commands

```bash
# 1. Install deps
pip install torch

# 2. Export TorchScript preproc module (required before training)
python -m preproc.export

# 3. Train (exports preproc automatically if missing)
python -m trainer.train --epochs 3 --steps 200 --batch 128

# 4. Or run both steps at once
./scripts/run_training.sh

# 5. Build C++ preproc_runner
./scripts/build_cpp.sh

# 6. Run C++ binary (after build + export)
./build/preproc_runner artifacts/preproc.pt --batch 128
```

## Preprocessing pipeline

### DensePreproc (module.py)
Vectorized ops, no Python loops, TorchScript-compatible:
1. **Selective log1p** — applied to features flagged in `log_mask` (bool buffer).
   Input shifted to ≥0 per feature first.
2. **Z-score normalization** — per-feature `mean` / `std` buffers fitted from data.
   Call `module.fit_dense(sample_tensor)` to populate.
3. **Clamp [-5, 5]** — outlier suppression.

### SparsePreproc (module.py)
1. **Modulo hashing** — maps raw IDs (possibly OOV) into `[0, vocab_size)` per feature.
   Vocab sizes stored in `vocab_sizes` buffer.

### FeaturePreproc (module.py)
Combines both. This is the exported module. Forward signature:
```python
def forward(
    dense: Tensor,                    # [B, D]
    sparse_indices: List[Tensor],     # [total_ids_i] per feature
    sparse_offsets: List[Tensor],     # [B] bag offsets per feature
) -> Tuple[Tensor, List[Tensor], List[Tensor]]
```

## Model architecture (model/dlrm.py)

Mirrors the TorchRec API exactly so switching to real TorchRec is a one-line import change.

```
dense_arch  (MLP): [13] → [512, 256, 64]       # project dense to emb_dim
sparse_arch (EBC): 26 × EmbeddingBag(vocab, 64) # pool sparse IDs
interaction:       pairwise dot-products on all (dense+sparse) embeddings
over_arch   (MLP): [interact_dim + 64] → [512, 256, 1]
loss:              BCEWithLogitsLoss
optimizer:         AdaGrad (standard for DLRM)
```

Key classes:
- `EmbeddingBagConfig` — mirrors `torchrec.EmbeddingBagConfig`
- `EmbeddingBagCollection` — mirrors `torchrec.modules.embedding_modules.EmbeddingBagCollection`
- `DLRM` — mirrors `torchrec.models.dlrm.DLRM`
- `DLRMTrain` — mirrors `torchrec.models.dlrm.DLRMTrain`; returns `(loss, logits)`

### Switching to real TorchRec (GPU/Linux)
```bash
pip install torchrec
```
Then replace `from model.dlrm import ...` with the equivalent torchrec imports.
The constructor signatures are identical.

## Data (data/synthetic.py)

`CriteoSynthetic` — IterableDataset mimicking Criteo Kaggle structure:
- 13 dense features: half Exponential(0.5), half Normal(0, 3)
- 26 sparse features: power-law ID distribution (heavy head)
- Binary label correlated with first dense feature

`collate_fn` — assembles per-sample lists into batched EmbeddingBag inputs.

## C++ runner (preproc_runner/main.cpp)

Loads the TorchScript `artifacts/preproc.pt` via `torch::jit::load`, generates
a synthetic batch using LibTorch random ops, runs the module, and prints timing
and output statistics. Build requires LibTorch (bundled with PyTorch install).

## TorchScript constraints to maintain

All code in `preproc/module.py` must stay TorchScript-compatible:
- **No `.bool()`** on tensors in forward — store bool buffers directly
  (`register_buffer("name", t.to(torch.bool))`)
- **`List[int] | None`** type hints are OK (Python 3.10+), but not inside forward methods
- **`enumerate`** for iterating `ModuleList` (required by TorchScript)
- **No dynamic shapes** in conditional branches

## Training timing (batch=128, CPU, 3 epochs × 200 steps)

```
preproc:  ~0.2–0.6 ms / step  (TorchScript module, vectorized)
model:    ~76–80   ms / step  (26 EmbeddingBag + dot-product interaction + MLP)
data:     ~32–33   ms / step  (synthetic generation + collate)
```

Preproc is <1% of step time — the embedding lookups dominate.

## Extending

### Add a new preprocessing op
1. Add to `DensePreproc.forward()` or `SparsePreproc.forward()` in `preproc/module.py`.
2. Keep TorchScript constraints (see above).
3. Re-export: `python -m preproc.export`
4. Rebuild C++: `./scripts/build_cpp.sh`

### Add real dataset (Criteo TSV)
Replace `make_dataloader()` in `trainer/train.py` with a real dataset reader.
The `collate_fn` signature is unchanged — just swap the source.

### GPU training
Pass `--device cuda` to `trainer/train.py`. The model and data moves are already
device-parameterized. On GPU, also set `--emb-dim 128` and larger `--batch`.
