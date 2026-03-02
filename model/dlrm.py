"""
model/dlrm.py
-------------
DLRM model mirroring the TorchRec API.

TorchRec availability:
  On Mac, torchrec/fbgemm_gpu cannot be installed. This module provides
  drop-in replacements that match TorchRec's public API:

    EmbeddingBagConfig  — config dataclass for one embedding table
    EmbeddingBagCollection — manages a list of EmbeddingBag tables
    DLRM                — the full DLRM architecture
    DLRMTrain           — DLRM + BCEWithLogitsLoss forward

  To swap in real TorchRec, replace this file's imports with:
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.models.dlrm import DLRM, DLRMTrain
    import torchrec

Architecture (faithful to Meta's open-source DLRM paper):
  dense_arch (MLP): [dense_in] → dense_arch_layers → emb_dim
  sparse_arch:       EmbeddingBag per sparse feature → [n_sparse, emb_dim]
  interaction:       cat(dense_out, sparse_outs) → dot-product interactions
  over_arch (MLP):   [interaction_dim] → over_arch_layers → 1
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# TorchRec-compatible config + collection
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingBagConfig:
    """Mirrors torchrec.EmbeddingBagConfig."""
    name:           str
    embedding_dim:  int
    num_embeddings: int
    feature_names:  List[str]
    pooling:        str = "sum"   # "sum" | "mean" | "max"


class EmbeddingBagCollection(nn.Module):
    """
    Mirrors torchrec.modules.embedding_modules.EmbeddingBagCollection.

    Manages one EmbeddingBag per config entry. Forward accepts parallel
    lists of (indices, offsets) — one per feature, in config order.
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.configs = tables
        self.embedding_bags = nn.ModuleList([
            nn.EmbeddingBag(
                num_embeddings=t.num_embeddings,
                embedding_dim=t.embedding_dim,
                mode=t.pooling,
            ).to(device)
            for t in tables
        ])

    def forward(
        self,
        sparse_indices: List[torch.Tensor],   # [total_i] per feature
        sparse_offsets: List[torch.Tensor],   # [B] per feature
    ) -> List[torch.Tensor]:
        """Return list of pooled embeddings, each [B, emb_dim]."""
        outs: List[torch.Tensor] = []
        for i, emb in enumerate(self.embedding_bags):
            outs.append(emb(sparse_indices[i], sparse_offsets[i]))
        return outs


# ──────────────────────────────────────────────────────────────────────────────
# MLP helper
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, layer_sizes: List[int], final_activation: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for i, dim in enumerate(layer_sizes):
        layers.append(nn.Linear(prev, dim))
        is_last = (i == len(layer_sizes) - 1)
        if not is_last or final_activation:
            layers.append(nn.ReLU())
        prev = dim
    return nn.Sequential(*layers)


# ──────────────────────────────────────────────────────────────────────────────
# DLRM
# ──────────────────────────────────────────────────────────────────────────────

class DLRM(nn.Module):
    """
    DLRM architecture (mirrors torchrec.models.dlrm.DLRM).

    Forward inputs:
      dense:          [B, dense_in_features]   — preprocessed dense
      sparse_indices: list of [total_ids_i]    — preprocessed sparse indices
      sparse_offsets: list of [B]              — EmbeddingBag bag offsets

    Forward output:
      logits: [B, 1]

    Architecture:
      dense_arch:  dense_in → dense_arch_layer_sizes → emb_dim
      sparse_arch: EmbeddingBagCollection → n_sparse × [B, emb_dim]
      interaction: cat(dense_emb, sparse_embs) then pairwise dot-products
                   → [B, interaction_dim]
      over_arch:   interaction_dim → over_arch_layer_sizes → 1
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],       # last entry must == emb_dim
        over_arch_layer_sizes: List[int],        # last entry must == 1
        emb_dim: Optional[int] = None,
    ):
        super().__init__()
        self.ebc = embedding_bag_collection
        n_sparse = len(embedding_bag_collection.configs)

        # Derive emb_dim from the last dense_arch layer
        _emb_dim = emb_dim or dense_arch_layer_sizes[-1]
        assert dense_arch_layer_sizes[-1] == _emb_dim, (
            "Last dense_arch layer must equal emb_dim so shapes align at interaction."
        )

        # Dense arch: project dense features to embedding space
        self.dense_arch = _mlp(dense_in_features, dense_arch_layer_sizes, final_activation=True)

        # Interaction: (n_sparse + 1) vectors of emb_dim
        # → pairwise dot products + dense concat
        n_interact = n_sparse + 1          # +1 for dense projection
        interact_dim = (n_interact * (n_interact + 1)) // 2  # upper-tri + diag
        over_in = interact_dim + _emb_dim  # concat interact features with dense

        # Over arch: final MLP → logit
        self.over_arch = _mlp(over_in, over_arch_layer_sizes, final_activation=False)

        self._emb_dim   = _emb_dim
        self._n_sparse  = n_sparse

    def forward(
        self,
        dense: torch.Tensor,
        sparse_indices: List[torch.Tensor],
        sparse_offsets: List[torch.Tensor],
    ) -> torch.Tensor:
        # Dense projection: [B, emb_dim]
        dense_emb = self.dense_arch(dense)

        # Sparse pooled embeddings: list of [B, emb_dim]
        sparse_embs = self.ebc(sparse_indices, sparse_offsets)

        # Stack all vectors: [B, n_interact, emb_dim]
        all_embs = torch.stack([dense_emb] + sparse_embs, dim=1)

        # Pairwise dot-product interaction: upper-triangle + diagonal
        # [B, n, d] @ [B, d, n] → [B, n, n]
        interact_mat = torch.bmm(all_embs, all_embs.transpose(1, 2))
        n = all_embs.shape[1]
        # Upper-triangular indices (including diagonal)
        rows, cols = torch.triu_indices(n, n, offset=0)
        interact_flat = interact_mat[:, rows, cols]   # [B, interact_dim]

        # Concat with dense projection and feed to over_arch
        x = torch.cat([dense_emb, interact_flat], dim=1)  # [B, over_in]
        return self.over_arch(x)                           # [B, 1]


# ──────────────────────────────────────────────────────────────────────────────
# DLRMTrain — adds loss to the forward pass (mirrors torchrec.models.dlrm.DLRMTrain)
# ──────────────────────────────────────────────────────────────────────────────

class DLRMTrain(nn.Module):
    """
    DLRM with BCEWithLogitsLoss baked into forward.
    Mirrors torchrec.models.dlrm.DLRMTrain.

    Returns (loss, logits) to match TorchRec's convention.
    """

    def __init__(self, dlrm: DLRM):
        super().__init__()
        self.model    = dlrm
        self.loss_fn  = nn.BCEWithLogitsLoss()

    def forward(
        self,
        dense: torch.Tensor,
        sparse_indices: List[torch.Tensor],
        sparse_offsets: List[torch.Tensor],
        labels: torch.Tensor,               # [B, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(dense, sparse_indices, sparse_offsets)
        loss   = self.loss_fn(logits, labels)
        return loss, logits


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dlrm(
    num_dense: int,
    vocab_sizes: List[int],
    emb_dim: int = 64,
    dense_arch: List[int] = None,
    over_arch:  List[int] = None,
    device: torch.device = torch.device("cpu"),
) -> DLRMTrain:
    """Convenience factory matching the TorchRec builder pattern."""
    if dense_arch is None:
        dense_arch = [512, 256, emb_dim]
    if over_arch is None:
        over_arch  = [512, 256, 1]

    tables = [
        EmbeddingBagConfig(
            name=f"t_{i}",
            embedding_dim=emb_dim,
            num_embeddings=v,
            feature_names=[f"sparse_{i}"],
        )
        for i, v in enumerate(vocab_sizes)
    ]
    ebc   = EmbeddingBagCollection(tables, device=device)
    dlrm  = DLRM(ebc, num_dense, dense_arch, over_arch)
    return DLRMTrain(dlrm).to(device)
