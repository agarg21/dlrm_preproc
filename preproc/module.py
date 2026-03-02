"""
preproc/module.py
-----------------
Feature preprocessing as TorchScript-compatible nn.Modules.

All learnable-like state is stored in register_buffer (not nn.Parameter) so:
  - no accidental gradient flow through preproc weights
  - TorchScript-compatible without modifications
  - safe to export with torch.jit.script() and load in C++

Pipeline:
  raw dense [B, D]     --DensePreproc-->  normed dense [B, D]
  raw sparse id-lists  --SparsePreproc--> hashed id-lists (same shape)
  combined             --FeaturePreproc-> (normed_dense, hashed_indices, offsets)
"""

from typing import List, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Dense preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class DensePreproc(nn.Module):
    """
    Vectorized dense feature preprocessing (TorchScript-compatible).

    Operations applied per batch (all element-wise, no Python loops):
      1. Selective log1p  — tames heavy-tailed count/frequency features.
         Applied only to features flagged in ``log_mask``.
         Input is first shifted to [0, ∞) per feature.
      2. Z-score normalization  — subtracts ``mean``, divides by ``std``.
         Statistics are fixed buffers set from a data pass (see fit()).
      3. Clamp to [-5, 5]  — prevents gradient explosion from outliers.

    Args:
        num_dense:   number of dense input features (D).
        log_features: indices of features to apply log1p to (default: all).
        eps:          numerical stability constant for std division.
    """

    def __init__(
        self,
        num_dense: int,
        log_features: List[int] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_dense = num_dense
        self.eps = eps
        self.clamp_lo: float = -5.0
        self.clamp_hi: float = 5.0

        # Per-feature log mask (1.0 = apply log1p, 0.0 = skip)
        if log_features is None:
            log_mask = torch.ones(num_dense)
        else:
            log_mask = torch.zeros(num_dense)
            for i in log_features:
                log_mask[i] = 1.0
        self.register_buffer("log_mask", log_mask.to(torch.bool))

        # Running statistics — updated by fit(), frozen during training
        self.register_buffer("mean", torch.zeros(num_dense))
        self.register_buffer("std",  torch.ones(num_dense))

    def fit(self, data: torch.Tensor) -> "DensePreproc":
        """Compute mean/std from a representative data sample (offline pass)."""
        with torch.no_grad():
            self.mean.copy_(data.mean(dim=0))
            self.std.copy_(data.std(dim=0).clamp(min=self.eps))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        # 1. Log1p on masked features (shift to non-negative first per-batch)
        shift = (x * self.log_mask).min(dim=0, keepdim=True).values.clamp(max=0.0)
        x_shifted = x - shift                          # all masked cols ≥ 0
        log_out = torch.log1p(x_shifted)               # safe: x_shifted ≥ 0
        x = torch.where(self.log_mask, log_out, x)

        # 2. Z-score normalise
        x = (x - self.mean) / (self.std + self.eps)

        # 3. Stability clamp
        x = torch.clamp(x, self.clamp_lo, self.clamp_hi)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Sparse preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class SparsePreproc(nn.Module):
    """
    Vectorized sparse ID preprocessing (TorchScript-compatible).

    Operations:
      1. Modulo hashing — maps raw IDs (possibly out-of-vocab) into
         [0, vocab_size) for each feature. Handles unseen / large IDs safely.
      2. (Future) frequency clipping — UNK-token for very rare IDs.

    Args:
        vocab_sizes: per-feature vocabulary sizes as a list.
    """

    def __init__(self, vocab_sizes: List[int]):
        super().__init__()
        self.register_buffer(
            "vocab_sizes", torch.tensor(vocab_sizes, dtype=torch.long)
        )

    def forward(self, indices: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Hash one sparse feature's raw indices into its vocab range."""
        vocab = int(self.vocab_sizes[feature_idx].item())
        return indices % vocab


# ──────────────────────────────────────────────────────────────────────────────
# Combined preprocessing (what gets TorchScript-exported and loaded in C++)
# ──────────────────────────────────────────────────────────────────────────────

class FeaturePreproc(nn.Module):
    """
    Combined dense + sparse feature preprocessing.

    This is the single module that is:
      - torch.jit.script()-ed in preproc/export.py
      - saved to artifacts/preproc.pt
      - loaded in C++ via torch::jit::load()
      - reused in Python training via torch.jit.load()

    Args:
        num_dense:   number of dense features.
        vocab_sizes: per-feature vocab sizes for sparse features.
        log_features: dense feature indices to apply log1p (default: all).
    """

    def __init__(
        self,
        num_dense: int,
        vocab_sizes: List[int],
        log_features: List[int] | None = None,
    ):
        super().__init__()
        self.dense  = DensePreproc(num_dense, log_features)
        self.sparse = SparsePreproc(vocab_sizes)
        self.num_sparse: int = len(vocab_sizes)

    def fit_dense(self, data: torch.Tensor) -> "FeaturePreproc":
        """Fit dense normalization stats from a data sample."""
        self.dense.fit(data)
        return self

    def forward(
        self,
        dense: torch.Tensor,                   # [B, num_dense]
        sparse_indices: List[torch.Tensor],    # list of [total_ids_i]
        sparse_offsets: List[torch.Tensor],    # list of [B]  (bag offsets)
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        # Dense path
        dense_out = self.dense(dense)

        # Sparse path — hash each feature independently
        hashed_indices: List[torch.Tensor] = []
        for i in range(len(sparse_indices)):
            hashed_indices.append(self.sparse(sparse_indices[i], i))

        return dense_out, hashed_indices, sparse_offsets
