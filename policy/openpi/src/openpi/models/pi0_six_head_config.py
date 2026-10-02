"""Configuration for a shared-VLM policy with six routed action experts.

PaliGemma provides a shared backbone; each action expert uses a Gemma tower.
Training routes each sample by skill_canonical and inference by stage_hint.
Each sample updates its selected expert and the trainable shared backbone."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_six_head import Pi0SixHead


CANONICAL_HEADS: tuple[str, ...] = (
    "move_to",
    "pick_up_from",
    "place_in",
    "place_on",
    "open",
    "close",
)


@dataclasses.dataclass(frozen=True)
class Pi0SixHeadConfig(pi0_config.Pi0Config):
    """6-expert π0.5 config.

    Inherits all of Pi0Config so that ModelTransformFactory and the existing
    config plumbing (default_prompt, action_dim, action_horizon, max_token_len,
    pi05/discrete_state_input branching, etc.) keep working unchanged.
    """

    # Fixed per-class names used for diagnostics + checkpoint metadata.
    expert_names: tuple[str, ...] = CANONICAL_HEADS

    # 32 to match the pi05_base checkpoint shape (b1k 4-stage uses 32 as well).
    # The R1Pro action is 23-dim; the data pipeline must pad with zeros to 32.
    action_dim: int = 32

    # Default action-chunk horizon; benchmark recipes may override it.
    action_horizon: int = 30

    # π0.5 mode (state goes through discrete language tokens, action expert uses adaRMS).
    pi05: bool = True

    # ----- training-side enhancements -----

    # Number of noise/time samples per flow-matching loss evaluation.
    num_flow_samples: int = 1

    # Correlated noise: sample noise from N(0, β·I + (1-β)·Σ) where Σ is the
    # per-expert action correlation matrix from per-expert norm_stats.
    # 1.0 = independent (default, no correlation). Recommended if using: 0.5.
    # Requires ``correlation_matrices_path`` to point at a directory containing
    # ``<expert_name>/norm_stats.json`` files (output of compute_norm_stats.py).
    correlation_beta: float = 1.0
    # Filesystem path to the directory holding 6 per-expert norm_stats.json
    # (each carrying a ``correlation_matrix`` field). When unset, correlated
    # noise is disabled regardless of ``correlation_beta``.
    correlation_matrices_path: str | None = None

    @property
    def num_action_experts(self) -> int:
        return len(self.expert_names)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        # Reuse PI05 — the public model type doesn't need to expose 6-head as a
        # new enum value; the 6-head dispatch is purely an internal arch detail.
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0SixHead":
        from openpi.models.pi0_six_head import Pi0SixHead

        return Pi0SixHead(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                # 6-head dispatch hint (one int per sample, value in [0, num_action_experts)).
                skill_canonical_ids=jax.ShapeDtypeStruct([batch_size], jnp.int32),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the vision encoder while training the language backbone and experts.

        LoRA variants use the base configuration's adapter filter. RoboCasa joint
        training overrides this filter with nnx.Nothing() to train the full VLM."""
        # LoRA case: defer to base filter so LoRA-adapter-only training works.
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            return super().get_freeze_filter()
        # Freeze only the SigLIP vision encoder.
        return nnx_utils.PathRegex(".*img.*")
