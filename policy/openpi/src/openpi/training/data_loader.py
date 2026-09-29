from collections import Counter
from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import platform
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.transforms_normalize import NormalizeWithPerTimestamp, PerExpertNormalize
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if data_config.behavior_manifest_path is not None:
        from openpi.training import task0_stage_training
        from openpi.training.behavior_segment_dataset import BehaviorSegmentDataset

        if data_config.behavior_dataset_root is None:
            raise ValueError("behavior_dataset_root is required for BEHAVIOR segment dataset loading.")
        prompt_resolver = (
            task0_stage_training.runtime_prompt_for_row
            if data_config.behavior_prompt_style == "task0_runtime"
            else None
        )
        return BehaviorSegmentDataset(
            dataset_root=data_config.behavior_dataset_root,
            manifest_path=data_config.behavior_manifest_path,
            action_horizon=action_horizon,
            frame_cache_root=data_config.behavior_frame_cache_root,
            video_tolerance_s=data_config.behavior_video_tolerance_s,
            runtime_stage_id=data_config.behavior_runtime_stage_id,
            task_index_filter=data_config.behavior_task_index_filter,
            prompt_resolver=prompt_resolver,
        )
    if data_config.skill_segments_dir is not None:
        from openpi.training.skill_segment_dataset import SkillSegmentDataset, CANONICAL_HEADS

        canonical_heads = data_config.skill_segments_canonical_heads or CANONICAL_HEADS
        return SkillSegmentDataset(
            skill_segments_dir=data_config.skill_segments_dir,
            action_horizon=action_horizon,
            random_window=data_config.skill_segments_random_window,
            seed=0,
            use_per_frame_state_delta=data_config.skill_segments_use_per_frame_state_delta,
            canonical_heads=canonical_heads,
            prompt_style=data_config.skill_segments_prompt_style,
            state_column=data_config.skill_segments_state_column,
            action_column=data_config.skill_segments_action_column,
        )

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Per-expert norm path (opt-in via DataConfig.use_per_expert_norm).
    # Requires per_expert_norm_stats to have been loaded by the data config.
    if (
        not skip_norm_stats
        and data_config.use_per_expert_norm
        and data_config.per_expert_norm_stats is not None
    ):
        logging.info(
            f"PerExpertNormalize active: {len(data_config.per_expert_norm_stats)} per-head stats sets"
        )
        normalize_step = PerExpertNormalize(
            data_config.per_expert_norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )
    else:
        normalize_step = NormalizeWithPerTimestamp(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalize_step,
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    if (
        not skip_norm_stats
        and data_config.use_per_expert_norm
        and data_config.per_expert_norm_stats is not None
    ):
        logging.info(
            f"PerExpertNormalize active (iterable): "
            f"{len(data_config.per_expert_norm_stats)} per-head stats sets"
        )
        normalize_step = PerExpertNormalize(
            data_config.per_expert_norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )
    else:
        normalize_step = NormalizeWithPerTimestamp(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            normalize_step,
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    raw_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    sampler = None
    batch_sampler = None
    if data_config.behavior_stage_balanced_sampling and hasattr(raw_dataset, "runtime_stage_ids"):
        runtime_stage_ids = raw_dataset.runtime_stage_ids
        if runtime_stage_ids:
            weights = make_task0_unified_stage_weights(runtime_stage_ids)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(raw_dataset),
                replacement=True,
            )

    if data_config.skill_segments_dir is not None:
        # 6-head plan: use StratifiedWeightedBatchSampler. Counts per expert
        # are either proportional to pool size (default — equalizes per-sample
        # exposure across heads) or uniform = skill_segments_per_expert.
        from openpi.training.skill_segment_dataset import load_sampler_weights_in_order
        from openpi.training.stratified_sampler import StratifiedWeightedBatchSampler

        if not hasattr(raw_dataset, "skill_canonical_ids") or not hasattr(raw_dataset, "sample_ids"):
            raise TypeError(
                "skill_segments_dir is set but the dataset does not expose "
                "`skill_canonical_ids` / `sample_ids`. Did you point at a non-"
                "SkillSegmentDataset source?"
            )
        weights_in_order = load_sampler_weights_in_order(
            data_config.skill_segments_sampler_weights_path, raw_dataset.sample_ids
        )
        if data_config.skill_segments_chunk_weighted:
            if not hasattr(raw_dataset, "window_counts"):
                raise TypeError(
                    "skill_segments_chunk_weighted is set but the dataset does not expose "
                    "`window_counts`. Did you point at a non-SkillSegmentDataset source?"
                )
            window_counts = np.asarray(raw_dataset.window_counts, dtype=np.float64)
            if window_counts.shape != (len(weights_in_order),):
                raise ValueError(
                    f"window_counts shape {window_counts.shape} does not match weights "
                    f"length {len(weights_in_order)}"
                )
            weights_in_order = (np.asarray(weights_in_order, dtype=np.float64) * window_counts).tolist()
            logging.info(
                "skill_segments sampler: chunk-weighted within head "
                "(window_count min=%.0f median=%.0f max=%.0f)",
                float(window_counts.min()),
                float(np.median(window_counts)),
                float(window_counts.max()),
            )
        if data_config.skill_segments_task_balanced_within_head:
            if not hasattr(raw_dataset, "task_indices"):
                raise TypeError(
                    "skill_segments_task_balanced_within_head is set but the dataset does not expose "
                    "`task_indices`. Did you point at a non-SkillSegmentDataset source?"
                )
            skill_ids = np.asarray(raw_dataset.skill_canonical_ids, dtype=np.int64)
            task_ids = np.asarray(raw_dataset.task_indices, dtype=np.int64)
            weights = np.asarray(weights_in_order, dtype=np.float64)
            if task_ids.shape != weights.shape:
                raise ValueError(
                    f"task_indices shape {task_ids.shape} does not match weights length {weights.shape}"
                )

            balanced = weights.copy()
            summary_parts: list[str] = []
            for eid in range(data_config.skill_segments_num_experts):
                head_mask = skill_ids == eid
                if not np.any(head_mask):
                    continue
                head_total = float(weights[head_mask].sum())
                tasks = np.unique(task_ids[head_mask])
                if head_total <= 0.0 or tasks.size == 0:
                    continue
                target_per_task = head_total / float(tasks.size)
                for tid in tasks:
                    group = head_mask & (task_ids == tid)
                    group_total = float(weights[group].sum())
                    if group_total <= 0.0:
                        continue
                    balanced[group] *= target_per_task / group_total
                after = [
                    float(balanced[head_mask & (task_ids == tid)].sum() / balanced[head_mask].sum())
                    for tid in tasks
                ]
                summary_parts.append(
                    f"head{eid}:tasks={tasks.tolist()} pct={[round(v * 100.0, 1) for v in after]}"
                )
            weights_in_order = balanced.tolist()
            logging.info(
                "skill_segments sampler: task-balanced within head active (%s)",
                "; ".join(summary_parts),
            )
        num_experts = data_config.skill_segments_num_experts
        if data_config.skill_segments_single_head_index is not None:
            # Single-head mode: ALL batch_size samples come from one head.
            # per_expert_counts = (0, ..., batch_size, ..., 0) with bs at slot k.
            head_idx = int(data_config.skill_segments_single_head_index)
            if not 0 <= head_idx < num_experts:
                raise ValueError(
                    f"single_head_index {head_idx} out of range [0, {num_experts})"
                )
            single_counts = [0] * num_experts
            single_counts[head_idx] = batch_size
            per_expert_arg: int | list[int] = single_counts
            logging.info(
                "skill_segments stratified sampler (single_head): head_idx=%d "
                "per_expert_counts=%s sum=%d",
                head_idx, per_expert_arg, sum(per_expert_arg),
            )
        elif data_config.skill_segments_per_expert_counts_override is not None:
            override = list(data_config.skill_segments_per_expert_counts_override)
            if len(override) != num_experts:
                raise ValueError(
                    f"per_expert_counts_override length {len(override)} != num_experts {num_experts}"
                )
            if sum(override) != batch_size:
                raise ValueError(
                    f"per_expert_counts_override sums to {sum(override)} but batch_size={batch_size}"
                )
            per_expert_arg = override
            logging.info(
                "skill_segments stratified sampler (override): per_expert_counts=%s sum=%d",
                per_expert_arg, sum(per_expert_arg),
            )
        elif data_config.skill_segments_per_expert_proportional:
            chunk_proportional = data_config.skill_segments_per_expert_chunk_proportional
            row_weights = None
            if chunk_proportional:
                if not hasattr(raw_dataset, "window_counts"):
                    raise TypeError(
                        "skill_segments_per_expert_chunk_proportional is set but the dataset "
                        "does not expose `window_counts`."
                    )
                row_weights = np.asarray(raw_dataset.window_counts, dtype=np.float64)
            per_expert_arg = StratifiedWeightedBatchSampler.proportional_per_expert(
                raw_dataset.skill_canonical_ids, batch_size=batch_size, num_experts=num_experts,
                row_weights=row_weights,
            )
            logging.info(
                "skill_segments stratified sampler (proportional, %s): per_expert_counts=%s sum=%d",
                "chunk-fair" if chunk_proportional else "segment",
                per_expert_arg, sum(per_expert_arg),
            )
        else:
            per_expert_arg = data_config.skill_segments_per_expert
            logging.info(
                "skill_segments stratified sampler (uniform): per_expert=%d × %d experts = batch %d",
                per_expert_arg, num_experts, per_expert_arg * num_experts,
            )
        batch_sampler = StratifiedWeightedBatchSampler(
            skill_canonical_ids=raw_dataset.skill_canonical_ids,
            sample_weights=weights_in_order,
            per_expert=per_expert_arg,
            num_experts=num_experts,
            num_batches=None,  # 1 epoch worth; the outer DataLoader loops via __iter__
            seed=seed,
        )

    dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if sampler is not None:
                raise NotImplementedError("behavior_stage_balanced_sampling is not supported with PyTorch distributed training")
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and batch_sampler is None and shuffle),  # Don't shuffle if using a sampler
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def make_task0_unified_stage_weights(runtime_stage_ids: Sequence[int]) -> np.ndarray:
    if not runtime_stage_ids:
        raise ValueError("runtime_stage_ids must not be empty")
    counts = Counter(runtime_stage_ids)
    return np.asarray([1.0 / counts[stage_id] for stage_id in runtime_stage_ids], dtype=np.float32)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        # Only meaningful for plain sequential batching. A custom sampler /
        # batch_sampler (e.g. StratifiedWeightedBatchSampler) draws indices
        # with replacement, so a batch larger than the row count is valid —
        # each row just yields multiple random windows.
        if sampler is None and batch_sampler is None and len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        # Multiprocessing route disabled by default on this host:
        #   - fork dead-locks (parent has JAX init'd; workers inherit held mutexes
        #     whose owner threads don't survive fork → first batch never delivered).
        #   - spawn dead-locks too (with 48 workers each re-importing the heavy
        #     openpi.training.config / model module, the index pipe between main
        #     and workers fills before workers finish startup → main blocks on
        #     pipe_write while workers block on do_sys_poll).
        # Threading wins here because cv2 / pyarrow / Pillow all release the GIL
        # during their C-extension I/O, so 8-16 threads give real parallelism
        # without any process-fork hazard. Set OPENPI_USE_TORCH_DATALOADER=1
        # to force-enable the legacy torch DataLoader path (e.g. for hosts
        # where threading lacks IO concurrency, or RLDS-style streaming).
        use_threaded = num_workers > 0 and os.environ.get("OPENPI_USE_TORCH_DATALOADER", "0") != "1"
        mp_context = "spawn" if num_workers > 0 else None

        generator = torch.Generator()
        generator.manual_seed(seed)
        if batch_sampler is not None:
            # batch_sampler is mutually exclusive with batch_size/shuffle/sampler/drop_last.
            if sampler is not None:
                raise ValueError("sampler and batch_sampler are mutually exclusive")
            if use_threaded:
                self._data_loader = _ThreadedBatchLoader(
                    dataset=dataset,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    collate_fn=_collate_fn,
                    worker_init_fn=_worker_init_fn,
                    prefetch_factor=2,
                )
                logging.info(
                    "TorchDataLoader: using ThreadedBatchLoader (num_workers=%d, prefetch=2). "
                    "Set OPENPI_USE_TORCH_DATALOADER=1 to revert to torch.utils.data.DataLoader.",
                    num_workers,
                )
                _has_threaded = True
            else:
                _has_threaded = False
                self._data_loader = torch.utils.data.DataLoader(
                    typing.cast(torch.utils.data.Dataset, dataset),
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    multiprocessing_context=mp_context,
                    persistent_workers=num_workers > 0,
                    collate_fn=_collate_fn,
                    worker_init_fn=_worker_init_fn,
                    generator=generator,
                    prefetch_factor=2 if num_workers > 0 else None,
                )
        else:
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
                sampler=sampler,
                num_workers=num_workers,
                multiprocessing_context=mp_context,
                persistent_workers=num_workers > 0,
                collate_fn=_collate_fn,
                worker_init_fn=_worker_init_fn,
                drop_last=True,
                generator=generator,
                prefetch_factor=8 if num_workers > 0 else None,
            )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


class _ThreadedBatchLoader:
    """Drop-in replacement for ``torch.utils.data.DataLoader`` that uses
    threads instead of subprocesses for prefetch.

    Why threads here: cv2.VideoCapture, pyarrow.parquet, and PIL all release
    Python's GIL during their C-extension I/O, so N threads give real
    parallel throughput on cv2/parquet read while the main thread runs JAX
    training. fork/spawn don't: fork dead-locks on JAX-held mutexes inherited
    by workers; spawn dead-locks on the index-pipe filling before workers
    finish re-importing openpi at 48-worker scale on this host. Threading
    has none of those: no fork, no pipe, no re-import — just an in-process
    ThreadPoolExecutor over a producer iterator.

    Implements the subset of torch DataLoader API that ``TorchDataLoader``
    actually uses: ``__iter__`` yielding batches, ``__len__`` for tqdm.
    """

    def __init__(
        self,
        *,
        dataset,
        batch_sampler,
        num_workers: int,
        collate_fn,
        worker_init_fn=None,
        prefetch_factor: int = 2,
    ) -> None:
        self._dataset = dataset
        self._batch_sampler = batch_sampler
        self._num_workers = max(1, int(num_workers))
        self._collate_fn = collate_fn
        self._prefetch_target = self._num_workers * max(1, int(prefetch_factor))
        if worker_init_fn is not None:
            worker_init_fn(0)  # main process — set XLA env once

    @property
    def batch_sampler(self):
        # Public alias matches torch.utils.data.DataLoader so train.py's
        # cursor walk (looks for `.batch_sampler` to extract per_expert_counts
        # from a StratifiedWeightedBatchSampler) finds it through this loader
        # too. Without this, per_expert_counts is None and the 6-head model
        # rejects non-uniform batches.
        return self._batch_sampler

    def _build_batch(self, indices):
        return self._collate_fn([self._dataset[i] for i in indices])

    def __iter__(self):
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        ex = ThreadPoolExecutor(max_workers=self._num_workers, thread_name_prefix="openpi_data")
        try:
            sampler_iter = iter(self._batch_sampler)
            in_flight: "deque" = deque()
            # Prime the queue.
            while len(in_flight) < self._prefetch_target:
                try:
                    in_flight.append(ex.submit(self._build_batch, next(sampler_iter)))
                except StopIteration:
                    break
            # Yield + replenish to keep ``_prefetch_target`` futures running.
            while in_flight:
                fut = in_flight.popleft()
                yield fut.result()
                try:
                    in_flight.append(ex.submit(self._build_batch, next(sampler_iter)))
                except StopIteration:
                    pass
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    def __len__(self) -> int:
        return len(self._batch_sampler)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
