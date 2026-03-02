"""
data/synthetic.py
-----------------
Synthetic Criteo-like DLRM dataset.

Mimics the Criteo Kaggle display-advertising dataset structure:
  - 13 dense (integer count) features — heavy-tailed, log-normally distributed
  - 26 sparse (categorical ID-list) features — power-law ID distribution
  - binary label (click / no-click)

No external data download required; everything is generated on-the-fly.
"""

from typing import Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, IterableDataset

# ── Criteo-inspired vocabulary sizes (capped at 200 000 for prototype) ───────
_CRITEO_VOCAB = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3,
    93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652,
    2173, 4, 7046547, 18, 15, 286181, 105, 142572,
]
VOCAB_SIZES: List[int] = [min(v, 200_000) for v in _CRITEO_VOCAB]
NUM_SPARSE: int = len(VOCAB_SIZES)
NUM_DENSE:  int = 13


def make_fit_sample(n: int, num_dense: int = NUM_DENSE) -> torch.Tensor:
    """Return a [n, num_dense] tensor of synthetic dense features for fitting stats."""
    # Mix of heavy-tailed count features (exp) and near-Gaussian features
    half = num_dense // 2
    left  = torch.distributions.Exponential(0.5).sample((n, half))
    right = torch.randn(n, num_dense - half) * 3.0
    return torch.cat([left, right], dim=1)


class CriteoSynthetic(IterableDataset):
    """
    Infinite IterableDataset of synthetic Criteo-like DLRM batches.

    Each item is a single sample (pre-batching done by DataLoader).

    Dense features:
      - First half: Exponential(0.5) — mimics count features (clicks, views…)
      - Second half: Normal(0, 3)    — mimics real-valued features

    Sparse features:
      - ID drawn from power-law ~ Uniform^2 * vocab_size * 2 (long tail)
      - Variable bag length: Uniform[1, 2*avg_len]

    Args:
        num_samples: total samples to yield per epoch (use large value for
                     effectively infinite streaming).
        avg_bag_len: average number of IDs per sparse feature per sample.
        vocab_sizes: per-feature vocabulary sizes.
        device:      torch device.
    """

    def __init__(
        self,
        num_samples: int = 100_000,
        avg_bag_len: int = 5,
        vocab_sizes: List[int] = VOCAB_SIZES,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_samples  = num_samples
        self.avg_bag_len  = avg_bag_len
        self.vocab_sizes  = vocab_sizes
        self.num_sparse   = len(vocab_sizes)
        self.device       = device

    def __iter__(self) -> Iterator[dict]:
        half = NUM_DENSE // 2
        for _ in range(self.num_samples):
            # Dense
            dense_left  = torch.distributions.Exponential(0.5).sample((half,))
            dense_right = torch.randn(NUM_DENSE - half) * 3.0
            dense = torch.cat([dense_left, dense_right])   # [D]

            # Sparse
            sparse_indices: List[torch.Tensor] = []
            sparse_lengths: List[int]          = []
            for vocab in self.vocab_sizes:
                length = torch.randint(1, self.avg_bag_len * 2 + 1, ()).item()
                # Power-law: square a uniform to get heavy head
                raw_ids = (torch.rand(length) ** 2 * vocab * 2).long()
                sparse_indices.append(raw_ids % vocab)
                sparse_lengths.append(length)

            # Label: sigmoid(sum of first dense feature) for some correlation
            prob   = torch.sigmoid(dense[0] / 5.0).item()
            label  = torch.tensor(float(torch.rand(()).item() < prob))

            yield {
                "dense":          dense,          # [D]
                "sparse_indices": sparse_indices, # list of [bag_len_i]
                "sparse_lengths": sparse_lengths, # list of int
                "label":          label,          # scalar
            }


def collate_fn(samples: List[dict]) -> Tuple[
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
]:
    """
    Collate a list of samples into a batch.

    Returns:
        dense:          [B, D]
        sparse_indices: list of [total_ids_i]  (concatenated per feature)
        sparse_offsets: list of [B]            (EmbeddingBag-style offsets)
        labels:         [B, 1]
    """
    B = len(samples)
    dense  = torch.stack([s["dense"] for s in samples])          # [B, D]
    labels = torch.tensor([[s["label"]] for s in samples])       # [B, 1]

    num_sparse = len(samples[0]["sparse_indices"])
    sparse_indices: List[torch.Tensor] = []
    sparse_offsets: List[torch.Tensor] = []

    for f in range(num_sparse):
        ids_list  = [s["sparse_indices"][f] for s in samples]
        offsets   = torch.zeros(B, dtype=torch.long)
        running   = 0
        for i, ids in enumerate(ids_list):
            offsets[i] = running
            running    += len(ids)
        sparse_indices.append(torch.cat(ids_list))
        sparse_offsets.append(offsets)

    return dense, sparse_indices, sparse_offsets, labels


def make_dataloader(
    batch_size: int = 128,
    num_samples: int = 100_000,
    avg_bag_len: int = 5,
    vocab_sizes: List[int] = VOCAB_SIZES,
    num_workers: int = 0,
) -> DataLoader:
    ds = CriteoSynthetic(
        num_samples=num_samples,
        avg_bag_len=avg_bag_len,
        vocab_sizes=vocab_sizes,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        # IterableDataset: don't shuffle (no __len__)
    )
