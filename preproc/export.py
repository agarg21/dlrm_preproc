"""
preproc/export.py
-----------------
Script the FeaturePreproc module and save it to artifacts/preproc.pt.

Usage:
    python -m preproc.export [--num-dense 13] [--out artifacts/preproc.pt]

The saved .pt file can be loaded by:
  - Python:  torch.jit.load("artifacts/preproc.pt")
  - C++:     torch::jit::load("artifacts/preproc.pt")
"""

import argparse
from pathlib import Path

import torch

from data.synthetic import VOCAB_SIZES, NUM_DENSE, make_fit_sample
from preproc.module import FeaturePreproc


def export(
    out_path: str = "artifacts/preproc.pt",
    num_dense: int = NUM_DENSE,
    vocab_sizes: list = VOCAB_SIZES,
    fit_batches: int = 20,
    batch_size: int = 128,
) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Build module
    module = FeaturePreproc(num_dense=num_dense, vocab_sizes=vocab_sizes)

    # Fit dense normalization on a representative data sample
    print(f"Fitting dense stats on {fit_batches} synthetic batches …")
    sample = make_fit_sample(fit_batches * batch_size, num_dense)
    module.fit_dense(sample)
    print(f"  mean range: [{module.dense.mean.min():.3f}, {module.dense.mean.max():.3f}]")
    print(f"  std  range: [{module.dense.std.min():.3f},  {module.dense.std.max():.3f}]")

    # TorchScript
    scripted = torch.jit.script(module)
    scripted.save(out_path)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"Saved TorchScript module → {out_path}  ({size_kb:.1f} KB)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-dense",  type=int, default=NUM_DENSE)
    p.add_argument("--out", default="artifacts/preproc.pt")
    p.add_argument("--fit-batches", type=int, default=20)
    args = p.parse_args()
    export(args.out, args.num_dense, VOCAB_SIZES, args.fit_batches)
