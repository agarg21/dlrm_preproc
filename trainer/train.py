"""
trainer/train.py
----------------
Full DLRM training harness.

Pipeline:
  DataLoader (CriteoSynthetic)
    → PreprocService (FeaturePreproc TorchScript module)
      → DLRMTrain forward
        → BCEWithLogitsLoss
          → AdaGrad optimizer step

PreprocService runs the *same* TorchScript .pt file that the C++ binary
(preproc_runner) also loads — ensuring Python training and C++ serving use
identical preprocessing logic.

Usage:
    # Export preproc first (one-time)
    python -m preproc.export

    # Then train
    python -m trainer.train [--epochs 3] [--steps 200] [--batch 128]
"""

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.synthetic import make_dataloader, VOCAB_SIZES, NUM_DENSE
from model.dlrm import build_dlrm
from preproc.export import export as export_preproc


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing service wrapper
# ──────────────────────────────────────────────────────────────────────────────

class PreprocService:
    """
    Wraps the TorchScript FeaturePreproc module for use in training.

    Loads the same .pt artifact that the C++ preproc_runner binary uses,
    ensuring preprocessing is 100% consistent between Python training and
    C++ serving.

    In production you would run this as a separate microservice process
    (e.g. reading from a Kafka queue, writing to shared memory). Here we
    run it in-process for training throughput, but the module boundary
    is kept clean: the trainer never touches raw data after this call.
    """

    def __init__(self, module_path: str):
        if not Path(module_path).exists():
            raise FileNotFoundError(
                f"Preproc module not found at {module_path}. "
                "Run `python -m preproc.export` first."
            )
        self.module = torch.jit.load(module_path)
        self.module.eval()

    @torch.no_grad()
    def __call__(
        self,
        dense:          torch.Tensor,
        sparse_indices: List[torch.Tensor],
        sparse_offsets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        return self.module(dense, sparse_indices, sparse_offsets)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

class RunningMean:
    def __init__(self): self.total = 0.0; self.n = 0
    def update(self, v, n=1): self.total += v * n; self.n += n
    @property
    def value(self): return self.total / max(self.n, 1)
    def reset(self): self.total = 0.0; self.n = 0


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits.sigmoid() >= 0.5).float()
    return (preds == labels).float().mean().item()


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    epochs:          int   = 3,
    steps_per_epoch: int   = 200,
    batch_size:      int   = 128,
    emb_dim:         int   = 64,
    dense_arch:      List[int] = None,
    over_arch:       List[int] = None,
    lr:              float = 0.01,
    preproc_path:    str   = "artifacts/preproc.pt",
    device_str:      str   = "cpu",
    log_every:       int   = 50,
) -> None:
    device = torch.device(device_str)

    # ── preprocessing service ─────────────────────────────────────────────────
    if not Path(preproc_path).exists():
        print(f"[setup] Preproc module not found — exporting to {preproc_path} …")
        export_preproc(preproc_path)
    preproc = PreprocService(preproc_path)
    print(f"[setup] Loaded preproc module from {preproc_path}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_dlrm(
        num_dense  = NUM_DENSE,
        vocab_sizes= VOCAB_SIZES,
        emb_dim    = emb_dim,
        dense_arch = dense_arch or [512, 256, emb_dim],
        over_arch  = over_arch  or [512, 256, 1],
        device     = device,
    )
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] Model parameters: {n_params:,}")

    # ── optimizer ─────────────────────────────────────────────────────────────
    # AdaGrad is standard for DLRM (handles sparse embedding grad updates well)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    # ── data ──────────────────────────────────────────────────────────────────
    loader = make_dataloader(
        batch_size  = batch_size,
        num_samples = steps_per_epoch * batch_size * epochs + batch_size,
    )
    data_iter = iter(loader)

    # ── training ──────────────────────────────────────────────────────────────
    print(f"\n[train] Starting: epochs={epochs}  steps/epoch={steps_per_epoch}  "
          f"batch={batch_size}  lr={lr}  device={device}\n")

    global_step = 0
    for epoch in range(1, epochs + 1):
        loss_meter = RunningMean()
        acc_meter  = RunningMean()
        t_data = t_preproc = t_model = 0.0

        for step in range(1, steps_per_epoch + 1):
            # ── fetch batch ───────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                dense, sp_idx, sp_off, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                dense, sp_idx, sp_off, labels = next(data_iter)
            dense   = dense.to(device)
            labels  = labels.to(device)
            sp_idx  = [t.to(device) for t in sp_idx]
            sp_off  = [t.to(device) for t in sp_off]
            t_data += time.perf_counter() - t0

            # ── preprocess (TorchScript module — same .pt as C++ binary) ─────
            t0 = time.perf_counter()
            dense_p, sp_idx_p, sp_off_p = preproc(dense, sp_idx, sp_off)
            t_preproc += time.perf_counter() - t0

            # ── forward + backward ────────────────────────────────────────────
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, logits = model(dense_p, sp_idx_p, sp_off_p, labels)
            loss.backward()
            optimizer.step()
            t_model += time.perf_counter() - t0

            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy(logits.detach(), labels), batch_size)
            global_step += 1

            if step % log_every == 0:
                print(
                    f"  epoch {epoch}/{epochs}  step {step:>4}/{steps_per_epoch}"
                    f"  loss={loss_meter.value:.4f}"
                    f"  acc={acc_meter.value:.3f}"
                    f"  [data={t_data*1e3/step:.1f}ms"
                    f"  preproc={t_preproc*1e3/step:.1f}ms"
                    f"  model={t_model*1e3/step:.1f}ms /step]"
                )

        print(
            f"[epoch {epoch}] loss={loss_meter.value:.4f}  acc={acc_meter.value:.3f}"
            f"  elapsed preproc={t_preproc*1e3:.0f}ms  model={t_model*1e3:.0f}ms"
        )

    print("\n[done] Training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLRM training harness")
    p.add_argument("--epochs",    type=int,   default=3)
    p.add_argument("--steps",     type=int,   default=200,   dest="steps_per_epoch")
    p.add_argument("--batch",     type=int,   default=128)
    p.add_argument("--emb-dim",   type=int,   default=64,    dest="emb_dim")
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--preproc",   default="artifacts/preproc.pt")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--log-every", type=int,   default=50,    dest="log_every")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs          = args.epochs,
        steps_per_epoch = args.steps_per_epoch,
        batch_size      = args.batch,
        emb_dim         = args.emb_dim,
        lr              = args.lr,
        preproc_path    = args.preproc,
        device_str      = args.device,
        log_every       = args.log_every,
    )
