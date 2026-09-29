"""Stratified-weighted batch sampler for the 6-head π0.5 plan.

What it does
------------
Each emitted batch has length ``batch_size = sum(per_expert_counts)``,
laid out as::

    [expert_0_idx]*c0 + [expert_1_idx]*c1 + ... + [expert_{N-1}_idx]*c_{N-1}

where ``ck`` is the per-expert sample count for slot k. Two modes:

- **Uniform** (``per_expert=int``): every expert contributes the same count.
  Each *expert* receives equal training; small-pool experts (e.g. close,
  n=804) end up over-repeating their samples relative to large-pool ones.
- **Proportional** (``per_expert=Sequence[int]`` from ``proportional_per_expert``):
  per-expert counts scale with pool size. Each *unique sample* is seen
  roughly the same number of times across all heads, at the cost of small
  experts receiving fewer total sample-views.

Within each expert's slot, indices are sampled with replacement using
per-sample weights (from ``sampler_weights.json``).

The 6-head model's ``compute_loss`` can statically slice each contiguous
slot and dispatch it to the matching expert — jit-clean, zero compute waste.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import logging

import numpy as np
import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class StratifiedWeightedBatchSampler(Sampler[list[int]]):
    """Yields batches of indices stratified by ``skill_canonical_id``.

    Args:
        skill_canonical_ids: per-row expert id (0..N-1), aligned with the
            dataset's row order. Length == len(dataset).
        sample_weights: per-row sampling weight, aligned with the dataset's
            row order. Length == len(dataset). Use the output of
            :func:`openpi.training.skill_segment_dataset.load_sampler_weights_in_order`.
        per_expert: number of samples per expert per batch. Total batch size
            = ``num_experts * per_expert``.
        num_experts: number of expert classes (default 6).
        num_batches: how many batches one epoch yields. Default
            ``len(dataset) // (num_experts * per_expert)`` to roughly match
            an epoch over the full dataset.
        seed: RNG seed for reproducibility (each rank should pass a distinct
            seed under DDP; or use ``DistributedSampler`` semantics by passing
            ``rank`` + ``world_size``).
    """

    def __init__(
        self,
        *,
        skill_canonical_ids: Sequence[int] | np.ndarray,
        sample_weights: Sequence[float] | np.ndarray,
        per_expert: int | Sequence[int],
        num_experts: int = 6,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        skill_canonical_ids = np.asarray(skill_canonical_ids, dtype=np.int64)
        sample_weights = np.asarray(sample_weights, dtype=np.float64)

        if skill_canonical_ids.shape != sample_weights.shape:
            raise ValueError(
                f"shape mismatch: skill_canonical_ids {skill_canonical_ids.shape} "
                f"vs sample_weights {sample_weights.shape}"
            )
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")

        # Normalize per_expert to a fixed-length list of ints (one per expert).
        if isinstance(per_expert, int):
            if per_expert < 1:
                raise ValueError(f"per_expert must be >= 1 when given as int, got {per_expert}")
            per_expert_list = [int(per_expert)] * num_experts
        else:
            per_expert_list = [int(c) for c in per_expert]
            if len(per_expert_list) != num_experts:
                raise ValueError(
                    f"per_expert sequence length {len(per_expert_list)} != num_experts {num_experts}"
                )
            if any(c < 0 for c in per_expert_list):
                raise ValueError(f"per-expert counts must be >= 0, got {per_expert_list}")
            if sum(per_expert_list) < 1:
                raise ValueError(f"per_expert counts sum to 0, got {per_expert_list}")

        # Pre-bucket indices + weights per expert. Using torch tensors so we
        # can use torch.multinomial (vectorized + stable seeding).
        self._per_expert_indices: list[torch.Tensor] = []
        self._per_expert_weights: list[torch.Tensor] = []
        for eid in range(num_experts):
            mask = skill_canonical_ids == eid
            ids = np.flatnonzero(mask)
            if ids.size == 0:
                raise ValueError(
                    f"expert id {eid} has zero samples — check skill_canonical_ids "
                    f"alignment with the dataset"
                )
            w = sample_weights[ids]
            if not np.all(w > 0):
                raise ValueError(
                    f"expert id {eid} has non-positive sample weights — check "
                    f"sampler_weights.json provenance"
                )
            self._per_expert_indices.append(torch.as_tensor(ids, dtype=torch.long))
            self._per_expert_weights.append(torch.as_tensor(w, dtype=torch.double))

        self._num_experts = int(num_experts)
        self._per_expert_counts: list[int] = list(per_expert_list)
        self._batch_size = sum(self._per_expert_counts)

        if num_batches is None:
            n = int(skill_canonical_ids.shape[0])
            num_batches = max(1, n // self._batch_size)
        self._num_batches = int(num_batches)
        self._seed = int(seed)
        self._epoch = 0

    # ----------------------------------------------------------- introspection

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def per_expert(self) -> list[int]:
        """Per-expert sample counts in slot order (one int per expert)."""
        return list(self._per_expert_counts)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def per_expert_pool_sizes(self) -> list[int]:
        """Number of unique samples in each expert's pool (after stratification)."""
        return [int(t.numel()) for t in self._per_expert_indices]

    # ----------------------------------------------------------- DDP support

    def set_epoch(self, epoch: int) -> None:
        """Advance the seed deterministically across epochs (matches
        ``DistributedSampler.set_epoch`` API)."""
        self._epoch = int(epoch)

    # ----------------------------------------------------------- iteration

    def __iter__(self) -> Iterator[list[int]]:
        # Deterministic per-epoch generator — same seed across workers if
        # DataLoader uses ``persistent_workers=False`` (default).
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch * 1_000_003)

        for _ in range(self._num_batches):
            batch: list[int] = []
            for eid in range(self._num_experts):
                count = self._per_expert_counts[eid]
                if count == 0:
                    continue  # single-head training: skip inactive slots entirely
                pool = self._per_expert_indices[eid]
                weights = self._per_expert_weights[eid]
                # multinomial(replacement=True) — supports both num_samples
                # > pool size and weighted sampling.
                picks = torch.multinomial(
                    weights, num_samples=count, replacement=True, generator=gen
                )
                batch.extend(int(pool[i].item()) for i in picks.tolist())
            yield batch

    def __len__(self) -> int:
        return self._num_batches

    # ----------------------------------------------------------- proportional helper

    @staticmethod
    def proportional_per_expert(
        skill_canonical_ids: Sequence[int] | np.ndarray,
        batch_size: int,
        num_experts: int = 6,
        row_weights: Sequence[float] | np.ndarray | None = None,
    ) -> list[int]:
        """Return per-expert counts proportional to pool size, summing to ``batch_size``.

        Each pool gets ``ceil(pool_size / total × batch_size)`` samples,
        rounded by largest-fractional-part to ensure the counts sum exactly
        to ``batch_size``. Every count is at least 1 (so no head is starved
        of gradient signal even when its pool is < batch_size / total of total).

        ``pool_size`` is the segment count per head by default. Pass
        ``row_weights`` (e.g. ``window_counts``) to make the allocation
        proportional to the summed weight per head instead — this is the
        chunk-fair allocation: every action chunk gets equal exposure
        regardless of which head (or how long its segment) it belongs to.

        Example
        -------
        >>> ids = np.array([0]*12621 + [1]*11671 + [2]*8394 + [3]*3207 + [4]*1610 + [5]*804)
        >>> proportional_per_expert(ids, batch_size=48, num_experts=6)
        [16, 15, 10, 4, 2, 1]
        """
        skill_canonical_ids = np.asarray(skill_canonical_ids, dtype=np.int64)
        if row_weights is None:
            pool_sizes = [float((skill_canonical_ids == eid).sum()) for eid in range(num_experts)]
        else:
            row_weights = np.asarray(row_weights, dtype=np.float64)
            if row_weights.shape != skill_canonical_ids.shape:
                raise ValueError(
                    f"row_weights shape {row_weights.shape} != skill_canonical_ids "
                    f"shape {skill_canonical_ids.shape}"
                )
            pool_sizes = [
                float(row_weights[skill_canonical_ids == eid].sum()) for eid in range(num_experts)
            ]
        total = sum(pool_sizes)
        if total == 0:
            raise ValueError("all expert pools are empty")
        if batch_size < num_experts:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_experts ({num_experts}) so every "
                f"head gets at least 1 sample per batch"
            )
        raw = [batch_size * ps / total for ps in pool_sizes]
        floors = [max(1, int(r)) for r in raw]
        diff = batch_size - sum(floors)
        if diff > 0:
            order = sorted(range(num_experts), key=lambda i: raw[i] - floors[i], reverse=True)
            for i in order[:diff]:
                floors[i] += 1
        elif diff < 0:
            order = sorted(range(num_experts), key=lambda i: raw[i] - floors[i])
            taken = 0
            for i in order:
                while floors[i] > 1 and taken < -diff:
                    floors[i] -= 1
                    taken += 1
                if taken >= -diff:
                    break
        if sum(floors) != batch_size:
            raise RuntimeError(
                f"could not distribute batch_size={batch_size} across pool_sizes={pool_sizes}; "
                f"got counts={floors}"
            )
        return floors
