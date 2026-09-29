import logging
import os
import pathlib
from typing import Any

import flax.traverse_util as _flax_tu
from flax import nnx as _nnx
import jax as _jax
import jax.numpy as jnp
import orbax.checkpoint as _ocp

import openpi.models.model as _model
import openpi.policies.policy as _policy
from openpi.policies.skill_segment_policy import StageHintToSkillCanonicalId
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training.transforms_normalize import NormalizeWithPerTimestamp
from openpi.training.transforms_normalize import PerExpertNormalize
from openpi.training.transforms_normalize import PerExpertUnnormalize
from openpi.training.transforms_normalize import UnnormalizeWithPerTimestamp
import openpi.transforms as transforms


def _restore_int_list_keys(params):
    """Round-trip NNX list nodes: orbax serializes them with str digit keys,
    but the model definition uses int keys (Python list indices). Convert
    digit-string segments back to int so check_pytree_equality passes.
    """
    flat = _flax_tu.flatten_dict(params)
    fixed = {
        tuple(int(p) if isinstance(p, str) and p.lstrip("-").isdigit() else p for p in path): v
        for path, v in flat.items()
    }
    return _flax_tu.unflatten_dict(fixed)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        # Inlined equivalent of train_config.model.load(restore_params(...)) with
        # an extra round-trip fix: orbax's ``intersect_trees`` re-stringifies int
        # dict keys (NNX represents Python lists as int-keyed dicts), so we
        # convert digit-string keys → int both before and after the intersect.
        raw = _restore_int_list_keys(
            _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )
        proto = _nnx.eval_shape(train_config.model.create, _jax.random.key(0))
        graphdef, state = _nnx.split(proto)
        trimmed = _ocp.transform_utils.intersect_trees(state.to_pure_dict(), raw)
        trimmed = _restore_int_list_keys(trimmed)
        state.replace_by_pure_dict(trimmed)
        model = _nnx.merge(graphdef, state)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    # Per-expert norm path (mirrors data_loader logic). When opted in and
    # per-expert stats are loaded, use PerExpertNormalize/PerExpertUnnormalize
    # which dispatch on data["skill_canonical_ids"]. Otherwise fall back to
    # the legacy combined NormalizeWithPerTimestamp.
    if data_config.use_per_expert_norm and data_config.per_expert_norm_stats is not None:
        logging.info(
            f"Policy: PerExpertNormalize/Unnormalize active "
            f"({len(data_config.per_expert_norm_stats)} per-head stats sets)"
        )
        normalize_input = PerExpertNormalize(
            data_config.per_expert_norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )
        unnormalize_output = PerExpertUnnormalize(
            data_config.per_expert_norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )
    else:
        normalize_input = NormalizeWithPerTimestamp(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )
        unnormalize_output = UnnormalizeWithPerTimestamp(
            norm_stats,
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm,
        )

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            StageHintToSkillCanonicalId(),
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            normalize_input,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            unnormalize_output,
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )