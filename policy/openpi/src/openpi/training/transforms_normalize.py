import dataclasses

import jax
import numpy as np

from openpi.shared import array_typing as at
from openpi.shared.normalize import NormStats
from openpi.transforms import DataDict, DataTransformFn, apply_tree, pad_to_dim


@dataclasses.dataclass(frozen=True)
class NormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    strict: bool = False
    use_per_timestamp: bool = False

    def __post_init__(self):
        if self.norm_stats is None or not self.use_quantiles:
            return
        for stats in jax.tree.leaves(self.norm_stats):
            if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            mean = stats.per_timestamp_mean[..., : x.shape[-2], : x.shape[-1]]
            std = stats.per_timestamp_std[..., : x.shape[-2], : x.shape[-1]]
            return (x - mean) / (std + 1e-6)

        mean = stats.mean[..., : x.shape[-1]]
        std = stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None

        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class PerExpertNormalize(DataTransformFn):
    """Per-sample, per-expert normalization for the 6-head plan.

    For each sample, looks up ``data["skill_canonical_ids"]`` (an int per
    sample, value in ``[0, num_experts)``) and applies that expert's
    normalization stats. The data tree (``state``, ``actions``, etc.) is
    normalized using the matching expert's stats — different samples in the
    same batch may use different stats sets.

    ``per_expert_stats`` is a list of ``num_experts`` dicts, each having
    ``state`` and ``actions`` keys mapping to ``NormStats``. Index = expert id.
    """

    per_expert_stats: list[dict[str, NormStats]] | None
    use_quantiles: bool = False
    strict: bool = False
    use_per_timestamp: bool = False

    def __post_init__(self):
        if self.per_expert_stats is None or not self.use_quantiles:
            return
        for one_set in self.per_expert_stats:
            for stats in jax.tree.leaves(one_set):
                if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                    raise ValueError("Quantile normalization requires q01 and q99 in per-expert norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.per_expert_stats is None:
            return data
        # Per-sample dispatch (this transform runs per-sample inside DataLoader).
        sid = data.get("skill_canonical_ids")
        if sid is None:
            raise ValueError(
                "PerExpertNormalize requires `skill_canonical_ids` in the sample dict; "
                "did SkillSegmentInputs run before normalization?"
            )
        eid = int(np.asarray(sid).reshape(-1)[0])
        if eid < 0 or eid >= len(self.per_expert_stats):
            raise ValueError(f"skill_canonical_ids={eid} out of range [0, {len(self.per_expert_stats)})")
        stats = self.per_expert_stats[eid]
        # Apply the same logic as NormalizeWithPerTimestamp but with the
        # selected expert's stats tree.
        return apply_tree(
            data,
            stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            mean = stats.per_timestamp_mean[..., : x.shape[-2], : x.shape[-1]]
            std = stats.per_timestamp_std[..., : x.shape[-2], : x.shape[-1]]
            return (x - mean) / (std + 1e-6)
        mean = stats.mean[..., : x.shape[-1]]
        std = stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        q01 = stats.q01[..., : x.shape[-1]]
        q99 = stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class PerExpertUnnormalize(DataTransformFn):
    """Inverse of :class:`PerExpertNormalize` for serving (denorm action chunks)."""

    per_expert_stats: list[dict[str, NormStats]] | None
    use_quantiles: bool = False
    use_per_timestamp: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if self.per_expert_stats is None:
            return data
        sid = data.get("skill_canonical_ids")
        if sid is None:
            raise ValueError("PerExpertUnnormalize requires `skill_canonical_ids` in the sample dict")
        eid = int(np.asarray(sid).reshape(-1)[0])
        stats = self.per_expert_stats[eid]
        return apply_tree(
            data,
            stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            mean = pad_to_dim(stats.per_timestamp_mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.per_timestamp_std, x.shape[-1], axis=-1, value=1.0)
            return x * (std[: x.shape[-2], :] + 1e-6) + mean[: x.shape[-2], :]
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        q01 = stats.q01
        q99 = stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            restored = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return np.concatenate([restored, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class UnnormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    use_per_timestamp: bool = False

    def __post_init__(self):
        if self.norm_stats is None or not self.use_quantiles:
            return
        for stats in jax.tree.leaves(self.norm_stats):
            if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            mean = pad_to_dim(stats.per_timestamp_mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.per_timestamp_std, x.shape[-1], axis=-1, value=1.0)
            return x * (std[: x.shape[-2], :] + 1e-6) + mean[: x.shape[-2], :]

        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None

        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            stats_dim = stats.per_timestamp_q01.shape[-1]
            q01 = stats.per_timestamp_q01[: x.shape[-2], :stats_dim]
            q99 = stats.per_timestamp_q99[: x.shape[-2], :stats_dim]
            if stats_dim < x.shape[-1]:
                # Inference time: model output is padded to model_action_dim
                # (e.g. 32) but stats only cover the first ``stats_dim`` (23)
                # — passthrough the padded dims unchanged.
                restored = (x[..., :stats_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                return np.concatenate([restored, x[..., stats_dim:]], axis=-1)
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

        q01 = stats.q01
        q99 = stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            restored = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return np.concatenate([restored, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
