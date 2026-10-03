"""Compute normalization statistics for a config."""

from collections import defaultdict
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.models.model as _model
import openpi.policies.behavior_policy as behavior_policy
import openpi.shared.normalize as normalize
from openpi.training.behavior_segment_dataset import ManifestRow, classify_task0_row
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _load_behavior_manifest_rows(manifest_path: str) -> list[dict]:
    with Path(manifest_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def filter_behavior_rows(rows: list[dict], runtime_stage_id: int | None, task_index: int | None = None) -> list[dict]:
    filtered = rows
    if task_index is not None:
        filtered = [row for row in filtered if int(row["task_index"]) == task_index]

    if runtime_stage_id is None:
        return filtered

    runtime_filtered: list[dict] = []
    for row in filtered:
        manifest_row = ManifestRow(**row)
        if classify_task0_row(manifest_row) == runtime_stage_id:
            runtime_filtered.append(row)

    if not runtime_filtered:
        raise ValueError(f"No manifest rows remain after filtering for runtime_stage_id={runtime_stage_id}")
    return runtime_filtered


def _behavior_episode_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _build_behavior_episode_meta(dataset_root: Path, task_indices: list[int]) -> dict[tuple[int, int], tuple[Path, int, int]]:
    episode_meta: dict[tuple[int, int], tuple[Path, int, int]] = {}
    global_offset = 0
    for task_index in task_indices:
        task_dir = dataset_root / "data" / f"task-{task_index:04d}"
        parquet_files = sorted(task_dir.glob("episode_*.parquet"), key=_behavior_episode_sort_key)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {task_dir}")
        for parquet_path in parquet_files:
            episode_index = _behavior_episode_sort_key(parquet_path)
            num_rows = pq.ParquetFile(parquet_path).metadata.num_rows
            episode_meta[(task_index, episode_index)] = (parquet_path, global_offset, num_rows)
            global_offset += num_rows
    return episode_meta


def compute_behavior_segment_norm_stats(config: _config.TrainConfig, data_config: _config.DataConfig, max_frames: int | None = None):
    if data_config.behavior_manifest_path is None or data_config.behavior_dataset_root is None:
        raise ValueError("behavior_manifest_path and behavior_dataset_root are required for BEHAVIOR fast-path stats.")

    rows = _load_behavior_manifest_rows(data_config.behavior_manifest_path)
    rows = filter_behavior_rows(
        rows,
        data_config.behavior_runtime_stage_id,
        data_config.behavior_task_index_filter,
    )
    if max_frames is not None:
        rows = rows[:max_frames]

    dataset_root = Path(data_config.behavior_dataset_root).expanduser().resolve()
    task_indices = sorted({int(row["task_index"]) for row in rows})
    episode_meta = _build_behavior_episode_meta(dataset_root, task_indices)

    grouped_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[(int(row["task_index"]), int(row["episode_index"]))].append(row)

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    per_timestamp_action_stats = (
        [normalize.RunningStats() for _ in range(config.model.action_horizon)]
        if data_config.use_per_timestamp_norm
        else None
    )
    horizon_offsets = np.arange(config.model.action_horizon, dtype=np.int64)

    progress = tqdm.tqdm(total=len(rows), desc="Computing stats")
    for episode_key, episode_rows in grouped_rows.items():
        parquet_path, global_start, num_rows = episode_meta[episode_key]
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)

        local_indices = np.asarray(
            [int(row["dataset_index"]) - global_start for row in episode_rows],
            dtype=np.int64,
        )
        if np.any(local_indices < 0) or np.any(local_indices >= num_rows):
            raise ValueError(f"Manifest row index is out of range for episode {episode_key}")

        state_batch = behavior_policy.extract_behavior_state(states[local_indices])
        future_indices = np.minimum(local_indices[:, None] + horizon_offsets[None, :], num_rows - 1)
        action_batch = actions[future_indices]

        stats["state"].update(state_batch)
        stats["actions"].update(action_batch)
        if per_timestamp_action_stats is not None:
            for timestep, running_stats in enumerate(per_timestamp_action_stats):
                running_stats.update(action_batch[:, timestep, :])
        progress.update(len(episode_rows))

    progress.close()
    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    if per_timestamp_action_stats is not None:
        per_timestamp_stats = [running_stats.get_statistics() for running_stats in per_timestamp_action_stats]
        action_stats = norm_stats["actions"]
        norm_stats["actions"] = normalize.NormStats(
            mean=action_stats.mean,
            std=action_stats.std,
            q01=action_stats.q01,
            q99=action_stats.q99,
            per_timestamp_mean=np.stack([item.mean for item in per_timestamp_stats], axis=0),
            per_timestamp_std=np.stack([item.std for item in per_timestamp_stats], axis=0),
            per_timestamp_q01=np.stack([item.q01 for item in per_timestamp_stats], axis=0),
            per_timestamp_q99=np.stack([item.q99 for item in per_timestamp_stats], axis=0),
        )
    return norm_stats


# ----- ProcessPool worker (top-level for picklability) ----------------------
def _process_parquet_for_skill_norm_stats(args):
    """Worker: read one parquet, dense-window each segment, return partial stats.

    Args (tuple, all picklable):
        parquet_path: str
        segment_data: list of (start_idx_30hz:int, end_idx_30hz:int, canon:str)
        delta_mask: np.ndarray | None
        horizon: int
        sample_fraction: float
        seed: int

    Returns dict { "combined": {"state":..., "actions":...},
                   "per_expert": {canon: {"state":..., "actions":...}} }
    where each leaf is {"sum","sumsq","min","max","count"} with float64 arrays.
    """
    parquet_path, segment_data, delta_mask, horizon, sample_fraction, seed, use_per_frame_state_delta = args

    table = pq.read_table(parquet_path, columns=["observation.state", "action"])
    states = (
        pa.compute.list_flatten(table["observation.state"])
        .to_numpy(zero_copy_only=False)
        .reshape(-1, 256)
        .astype(np.float32, copy=False)
    )
    actions_full = (
        pa.compute.list_flatten(table["action"])
        .to_numpy(zero_copy_only=False)
        .reshape(-1, 23)
        .astype(np.float32, copy=False)
    )
    n_rows = states.shape[0]

    rng = np.random.RandomState(seed)
    horizon_offsets = np.arange(horizon, dtype=np.int64)

    starts_chunks: list[np.ndarray] = []
    ends_chunks: list[np.ndarray] = []
    canon_chunks: list[np.ndarray] = []
    for s_idx, e_idx, canon in segment_data:
        seg_start = max(0, min(int(s_idx), n_rows - 1))
        seg_end = max(seg_start, min(int(e_idx), n_rows))
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue
        if seg_len <= horizon:
            seg_starts = np.array([seg_start], dtype=np.int64)
        else:
            seg_starts = np.arange(seg_start, seg_end - horizon + 1, dtype=np.int64)
            if sample_fraction < 1.0 and seg_starts.size > 1:
                n_keep = max(1, int(seg_starts.size * sample_fraction))
                if n_keep < seg_starts.size:
                    sel = rng.choice(seg_starts.size, n_keep, replace=False)
                    seg_starts = np.sort(seg_starts[sel])
        starts_chunks.append(seg_starts)
        ends_chunks.append(np.full(seg_starts.size, seg_end - 1, dtype=np.int64))
        canon_chunks.append(np.full(seg_starts.size, canon, dtype=object))

    if not starts_chunks:
        return None
    starts = np.concatenate(starts_chunks)
    canons = np.concatenate(canon_chunks)

    idxs = np.minimum(starts[:, None] + horizon_offsets[None, :], np.concatenate(ends_chunks)[:, None])
    state_batch = behavior_policy.extract_behavior_state(states[starts])  # (n, 23)
    action_batch = actions_full[idxs]  # (n, H, 23)

    if delta_mask is not None:
        d = delta_mask.shape[-1]
        action_batch = action_batch.copy()
        if use_per_frame_state_delta:
            state_per_frame = behavior_policy.extract_behavior_state(states[idxs.reshape(-1)]).reshape(
                idxs.shape[0], horizon, -1
            )
            action_batch[..., :d] -= np.where(delta_mask, state_per_frame[..., :d], 0.0)
        else:
            sub = np.where(delta_mask, state_batch[:, :d], 0.0)
            action_batch[..., :d] -= sub[:, None, :]

    def _agg(b2d: np.ndarray) -> dict:
        b64 = b2d.astype(np.float64)
        return {
            "sum": b64.sum(axis=0),
            "sumsq": (b64 ** 2).sum(axis=0),
            "min": b2d.min(axis=0).astype(np.float64),
            "max": b2d.max(axis=0).astype(np.float64),
            "count": int(b2d.shape[0]),
        }

    actions_flat = action_batch.reshape(-1, action_batch.shape[-1])
    out: dict = {
        "combined": {
            "state": _agg(state_batch),
            "actions": _agg(actions_flat),
            "actions_pt": [_agg(action_batch[:, t, :]) for t in range(horizon)],
        },
        "per_expert": {},
    }
    for name in np.unique(canons):
        mask = canons == name
        sb = state_batch[mask]
        ab_full = action_batch[mask]  # (k, H, 23)
        ab = ab_full.reshape(-1, action_batch.shape[-1])
        out["per_expert"][str(name)] = {
            "state": _agg(sb),
            "actions": _agg(ab),
            "actions_pt": [_agg(ab_full[:, t, :]) for t in range(horizon)],
        }
    return out


def _merge_partial(g: dict, p: dict) -> None:
    """Merge partial stats dict p into running global g."""
    if g["count"] == 0:
        g["sum"] = p["sum"].copy()
        g["sumsq"] = p["sumsq"].copy()
        g["min"] = p["min"].copy()
        g["max"] = p["max"].copy()
        g["count"] = p["count"]
    else:
        g["sum"] += p["sum"]
        g["sumsq"] += p["sumsq"]
        g["min"] = np.minimum(g["min"], p["min"])
        g["max"] = np.maximum(g["max"], p["max"])
        g["count"] += p["count"]


def _finalize_norm_stats(g: dict, pt: list[dict] | None = None) -> normalize.NormStats:
    """Convert running aggregates to NormStats. q01/q99 use min/max.

    If pt provided (list of per-timestep agg dicts, one per horizon t),
    also fills per_timestamp_mean/std/q01/q99 fields.
    """
    n = g["count"]
    mean = (g["sum"] / n).astype(np.float32)
    var = g["sumsq"] / n - (g["sum"] / n) ** 2
    std = np.sqrt(np.maximum(0, var)).astype(np.float32)

    pt_mean = pt_std = pt_q01 = pt_q99 = None
    if pt is not None and len(pt) > 0 and pt[0]["count"] >= 2:
        pt_means = np.stack([(d["sum"] / d["count"]) for d in pt], axis=0)
        pt_vars = np.stack(
            [d["sumsq"] / d["count"] - (d["sum"] / d["count"]) ** 2 for d in pt],
            axis=0,
        )
        pt_mean = pt_means.astype(np.float32)
        pt_std = np.sqrt(np.maximum(0, pt_vars)).astype(np.float32)
        pt_q01 = np.stack([d["min"] for d in pt], axis=0).astype(np.float32)
        pt_q99 = np.stack([d["max"] for d in pt], axis=0).astype(np.float32)

    return normalize.NormStats(
        mean=mean,
        std=std,
        q01=g["min"].astype(np.float32),
        q99=g["max"].astype(np.float32),
        per_timestamp_mean=pt_mean,
        per_timestamp_std=pt_std,
        per_timestamp_q01=pt_q01,
        per_timestamp_q99=pt_q99,
    )


def compute_skill_segments_norm_stats(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    max_frames: int | None,
    *,
    compute_correlation: bool = False,
    compute_per_timestamp: bool = True,
    sample_fraction: float = 0.1,
) -> dict[str, normalize.NormStats] | None:
    """Compute (combined + per-expert) norm stats for the 6-head pipeline.

    Side-effect: writes ``per_expert/<expert_name>/norm_stats.json`` for
    each of the 6 canonical heads (consumed by ``PerExpertNormalize`` at
    training time) and returns the *combined* stats (kept as a fallback /
    diagnostic).

    Flags:
      - ``compute_correlation``: also save per-expert action correlation
        matrix Σ (consumed by correlated-noise sampling). Expensive; off by
        default since ``correlation_beta=1.0`` is the default model config.
      - ``compute_per_timestamp``: also save per-timestamp action stats
        (mean/std/q01/q99 for each of the H action chunk steps). Used by
        ``NormalizeWithPerTimestamp`` / ``PerExpertNormalize`` when
        ``use_per_timestamp_norm=True``. On by default.
    """
    from openpi.training.skill_segment_dataset import (
        CANONICAL_HEADS,
        SkillSegmentDataset,
    )

    if compute_correlation or compute_per_timestamp:
        print(
            f"WARNING: parallel skill_segments path does not yet compute "
            f"correlation/per_timestamp (compute_correlation={compute_correlation}, "
            f"compute_per_timestamp={compute_per_timestamp}); only mean/std/min/max.",
            flush=True,
        )

    skill_dir = Path(data_config.skill_segments_dir).expanduser().resolve()
    horizon = config.model.action_horizon
    print(f"[norm_stats] horizon={horizon}, skill_dir={skill_dir}", flush=True)

    # Sample sliding windows within segment bounds, subsample per segment,
    # and process parquet files in parallel.
    SAMPLE_FRACTION = sample_fraction
    print(f"[norm_stats] sample_fraction={SAMPLE_FRACTION} ({'FULL' if SAMPLE_FRACTION >= 1.0 else 'subsampled'})", flush=True)
    # Use the configured expert order, falling back to the default.
    canonical_heads = data_config.skill_segments_canonical_heads or CANONICAL_HEADS
    print(f"[norm_stats] canonical_heads = {canonical_heads}", flush=True)
    print(f"[norm_stats] constructing SkillSegmentDataset...", flush=True)
    dataset = SkillSegmentDataset(
        skill_segments_dir=str(skill_dir), action_horizon=horizon, random_window=False, seed=0,
        canonical_heads=canonical_heads,
    )
    rows = dataset.rows
    print(f"[norm_stats] dataset has {len(rows)} rows", flush=True)
    if max_frames is not None and max_frames < len(rows):
        rows = rows[:max_frames]

    # Delta mask must match training-time transform.
    use_delta = bool(getattr(data_config, "skill_segments_dir", None)) and \
                bool(getattr(config.data, "use_delta_joint_actions", True))
    if use_delta:
        delta_mask = np.asarray(transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1))
    else:
        delta_mask = None
    use_per_frame_state_delta = bool(getattr(data_config, "skill_segments_use_per_frame_state_delta", False))

    # Group rows by parquet path → one worker job per parquet.
    print(f"[norm_stats] grouping {len(rows)} rows by parquet...", flush=True)
    grouped: dict[str, list] = defaultdict(list)
    for row in rows:
        grouped[row.parquet].append(row)
    print(f"[norm_stats] grouped into {len(grouped)} parquet files", flush=True)

    # Build worker args (pickle-friendly tuples).
    args_list = []
    for parquet_path, ep_rows in grouped.items():
        seg_data = [(int(r.start_idx_30hz), int(r.end_idx_30hz), str(r.skill_canonical)) for r in ep_rows]
        seed = abs(hash(parquet_path)) & 0xffffffff
        args_list.append((parquet_path, seg_data, delta_mask, horizon, SAMPLE_FRACTION, seed, use_per_frame_state_delta))
    print(f"[norm_stats] built args_list with {len(args_list)} entries", flush=True)

    # Process one parquet at a time to bound memory usage.
    use_parallel = False
    num_workers = min(mp.cpu_count(), max(1, len(args_list) // 2))
    print(
        f"Computing norm stats: {len(args_list)} parquets, "
        f"{'parallel ' + str(num_workers) + ' workers' if use_parallel else 'sequential'}, "
        f"sample_fraction={SAMPLE_FRACTION}",
        flush=True,
    )

    # Aggregators (per-timestamp uses list of H slot dicts).
    def _empty():
        return {"sum": None, "sumsq": None, "min": None, "max": None, "count": 0}
    combined_g: dict[str, object] = {
        "state": _empty(),
        "actions": _empty(),
        "actions_pt": [_empty() for _ in range(horizon)],
    }
    per_expert_g: dict[str, dict] = {
        name: {
            "state": _empty(),
            "actions": _empty(),
            "actions_pt": [_empty() for _ in range(horizon)],
        }
        for name in canonical_heads
    }

    def _consume(partial):
        if partial is None:
            return
        _merge_partial(combined_g["state"], partial["combined"]["state"])
        _merge_partial(combined_g["actions"], partial["combined"]["actions"])
        for t, pt in enumerate(partial["combined"]["actions_pt"]):
            _merge_partial(combined_g["actions_pt"][t], pt)
        for name, pe in partial["per_expert"].items():
            if name not in per_expert_g:
                continue
            _merge_partial(per_expert_g[name]["state"], pe["state"])
            _merge_partial(per_expert_g[name]["actions"], pe["actions"])
            for t, pt in enumerate(pe["actions_pt"]):
                _merge_partial(per_expert_g[name]["actions_pt"][t], pt)

    progress = tqdm.tqdm(
        total=len(args_list), desc="parquets",
        mininterval=2.0, file=sys.stdout, ascii=True,
    )
    if use_parallel:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_parquet_for_skill_norm_stats, a) for a in args_list]
            for fut in as_completed(futures):
                _consume(fut.result())
                progress.update(1)
    else:
        for a in args_list:
            _consume(_process_parquet_for_skill_norm_stats(a))
            progress.update(1)
    progress.close()

    # Save per-expert stats.
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("asset_id or repo_id must be set to write skill_segments stats.")
    base_out = config.assets_dirs / asset_id
    base_out.mkdir(parents=True, exist_ok=True)
    for name in canonical_heads:
        if per_expert_g[name]["actions"]["count"] < 2:
            continue
        per_expert_norm = {
            "state": _finalize_norm_stats(per_expert_g[name]["state"]),
            "actions": _finalize_norm_stats(
                per_expert_g[name]["actions"], pt=per_expert_g[name]["actions_pt"]
            ),
        }
        out_dir = base_out / "per_expert" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        normalize.save(out_dir, per_expert_norm)
        print(f"  wrote per-expert stats: {out_dir}  (n_actions={per_expert_g[name]['actions']['count']})")

    # Combined fallback stats (with per_timestamp).
    if combined_g["actions"]["count"] < 2:
        return None
    return {
        "state": _finalize_norm_stats(combined_g["state"]),
        "actions": _finalize_norm_stats(combined_g["actions"], pt=combined_g["actions_pt"]),
    }


def main(
    config_name: str,
    max_frames: int | None = None,
    *,
    compute_correlation: bool = False,
    sample_fraction: float = 0.1,
):
    """Compute normalization statistics.

    Args:
        config_name: Training configuration name, such as ``mobiagent_behavior``.
        max_frames: Optional cap on segments processed.
        compute_correlation: Also compute per-expert action correlation matrices
            for correlated-noise sampling when ``model.correlation_beta < 1.0``.
            Available for skill-segment datasets; disabled by default.
        sample_fraction: Fraction of chunk windows sampled per segment.
            Defaults to 0.1; use 1.0 to include all windows."""
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.skill_segments_dir is not None:
        norm_stats = compute_skill_segments_norm_stats(
            config, data_config, max_frames,
            compute_correlation=compute_correlation,
            sample_fraction=sample_fraction,
        )
    elif data_config.behavior_manifest_path is not None:
        norm_stats = compute_behavior_segment_norm_stats(config, data_config, max_frames)
    elif data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )
        keys = ["state", "actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise ValueError("Either asset_id or repo_id must be set to write normalization stats.")
    output_path = config.assets_dirs / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
