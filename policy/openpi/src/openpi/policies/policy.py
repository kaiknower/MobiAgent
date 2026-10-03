from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup. ``expert_id`` is marked static so JIT specializes
            # one compiled version per expert (Pi0SixHead has 6) — branches
            # inside sample_actions key off this static int and would otherwise
            # trip ConcretizationTypeError on a traced array.
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("expert_id",)
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Rolling chunk inpainting.
        # Client sends ``prev_actions`` in *physical* action space — same
        # shape/units as the actions it last received. Shape (T, action_dim_user)
        # where action_dim_user is the user-facing dim (23 for our pipeline).
        # ``prev_actions_mask`` is a (action_horizon,) bool indicating which
        # positions in the new chunk to anchor (typically [0:T]).
        #
        # We re-run the input pipeline on a copy of obs that has ``actions``
        # set to ``prev_actions`` so the same DeltaActions /
        # NormalizeWithPerTimestamp[:T] / PadStatesAndActions stack that
        # processed training actions also normalizes the inpaint anchors —
        # crucial when ``use_per_timestamp_norm=True``, since slot t=0 and
        # slot t=26 have different stats.
        prev_actions_physical = obs.get("prev_actions") if isinstance(obs, dict) else None
        prev_actions_mask = obs.get("prev_actions_mask") if isinstance(obs, dict) else None
        if prev_actions_physical is not None:
            obs = {k: v for k, v in obs.items() if k not in ("prev_actions", "prev_actions_mask")}

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if not self._is_pytorch_model and "skill_canonical_ids" in inputs:
            # Hoist the per-request expert id out of the jitted graph so the
            # model's per-sample dispatch can branch in Python (static under jit).
            sample_kwargs["expert_id"] = int(np.asarray(inputs["skill_canonical_ids"]).reshape(-1)[0])
        if not self._is_pytorch_model and prev_actions_physical is not None:
            # Build a parallel batch where ``actions`` = prev_actions, run it
            # through the same input pipeline. Output ``actions`` is the
            # normalized + delta-encoded + padded form at slot [0:T] stats.
            # np.array(..., copy=True) so DeltaActions's in-place mutation
            # doesn't fail on msgpack-numpy's read-only deserialized arrays.
            prev_obs = {**obs, "actions": np.array(prev_actions_physical, copy=True)}
            prev_transformed = self._input_transform(prev_obs)
            prev_normalized = np.asarray(prev_transformed["actions"])  # (T, action_dim=32) post-pad
            ah = self._model.action_horizon
            ad = self._model.action_dim
            prev_full = np.zeros((ah, ad), dtype=prev_normalized.dtype)
            t_used = min(prev_normalized.shape[0], ah)
            prev_full[:t_used] = prev_normalized[:t_used]
            sample_kwargs["prev_actions"] = jnp.asarray(prev_full)[np.newaxis, ...]
            mask_arr = np.asarray(prev_actions_mask, dtype=bool).reshape(-1)
            mask_full = np.zeros(ah, dtype=bool)
            mask_full[: min(mask_arr.shape[0], ah)] = mask_arr[: min(mask_arr.shape[0], ah)]
            sample_kwargs["prev_actions_mask"] = jnp.asarray(mask_full)
            # Diagnostic: confirm rolling inpaint is actually engaged on this
            # chunk. Logs the input shape, how many anchor positions are
            # active, and the L2 norm of the prev_actions tail so we can spot
            # all-zero / nan inputs quickly.
            try:
                _t_in = np.asarray(prev_actions_physical)
                _l2 = float(np.linalg.norm(_t_in.reshape(-1)))
            except Exception:
                _l2 = float("nan")
            logging.info(
                "[inpaint] chunk has prev_actions: in=%s mask_active=%d/%d L2=%.3f",
                tuple(np.asarray(prev_actions_physical).shape),
                int(mask_full.sum()),
                ah,
                _l2,
            )
        else:
            if not self._is_pytorch_model:
                logging.info("[inpaint] no prev_actions (cold start / first chunk / post-reset)")
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if "skill_canonical_ids" in inputs:
            # Denormalization must use the same expert as action sampling.
            outputs["skill_canonical_ids"] = inputs["skill_canonical_ids"]
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
