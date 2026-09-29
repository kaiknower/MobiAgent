"""π0.5 with 6 parallel action expert towers (heavy variant).

Architecture
------------
- Shared PaliGemma backbone (~3B): SigLIP vision + gemma_2b language.
- 6 parallel action experts (gemma_300m each, ~300M): one per canonical skill.
- Per-expert input/output projections (6 sets of action_in_proj, action_out_proj,
  time_mlp_in, time_mlp_out).
- Total params: ~3B + 6 × 300M = ~4.8B.

Phase-2 enhancements (gated by config flags)
--------------------------------------------
- Multi-step flow matching (``num_flow_samples > 1``): prefix VLM runs once,
  the action expert is unrolled N times over different (noise, time) tuples
  reusing the prefix kv-cache. Reduces flow-matching gradient variance ~√N.
- Correlated noise (``correlation_beta < 1.0`` + ``correlation_matrices_path``):
  noise per sample is ``L_eid @ z`` where ``z ~ N(0, I)`` and ``L_eid`` is the
  Cholesky factor of ``β·I + (1-β)·Σ_eid``. Shapes the noise to respect the
  per-expert action-chunk correlation structure measured in training data.

Training-time dispatch (approach B from the design doc)
--------------------------------------------------------
- The data loader produces *stratified* batches: each batch has equal counts
  per expert in slot order (k = batch_size / num_action_experts samples each).
  ``compute_loss`` assumes ``observation.skill_canonical_ids`` looks like::

      [0]*k + [1]*k + [2]*k + [3]*k + [4]*k + [5]*k

  (or any permutation of slots so long as samples for the same expert are
  contiguous and the count per expert is uniform). Within each train step we
  run gemma 6 times — once per expert with the other 5 expert lanes set to
  ``None`` so they contribute zero compute and zero gradient. Per-sample one
  expert + the shared backbone receive gradient. Static shapes per sub-call →
  jit-clean.

If a future caller provides a fully random (non-stratified) batch, the
  ``_check_stratified_batch`` invariant will raise — by design, to surface the
  upstream sampler bug rather than silently masking gradients.
"""
from __future__ import annotations

import logging
from pathlib import Path

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_six_head_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

logger = logging.getLogger("openpi")


# Re-use the prefix-LM attention helpers verbatim from the single-expert pi0.
# Keep them local to avoid an import cycle on Pi0.
def make_attn_mask(input_mask, mask_ar):
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij", pos, 1.0 / period * 2 * jnp.pi, precision=jax.lax.Precision.HIGHEST
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0SixHead(_model.BaseModel):
    """π0.5 with N parallel action expert towers (default N=6)."""

    def __init__(self, config: pi0_six_head_config.Pi0SixHeadConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.num_action_experts = int(config.num_action_experts)
        self.expert_names = tuple(config.expert_names)
        self.num_flow_samples = int(config.num_flow_samples)
        self.correlation_beta = float(config.correlation_beta)
        if self.num_action_experts < 1:
            raise ValueError("num_action_experts must be >= 1")
        if len(self.expert_names) != self.num_action_experts:
            raise ValueError(
                f"expert_names length {len(self.expert_names)} != num_action_experts {self.num_action_experts}"
            )

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # gemma with 1 backbone + N action expert towers (all share self-attention).
        all_configs = [paligemma_config] + [action_expert_config] * self.num_action_experts
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=all_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        # PaliGemma slot uses no adaRMS; each action expert uses adaRMS iff pi05.
        use_adarms = [False] + [config.pi05] * self.num_action_experts
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)

        # SigLIP vision encoder (shared across all experts).
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # Per-expert IO layers.
        # Plain Python lists; nnx tracks them as pytree leaves automatically.
        self.action_in_projs = [
            nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            for _ in range(self.num_action_experts)
        ]
        self.action_out_projs = [
            nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
            for _ in range(self.num_action_experts)
        ]
        if config.pi05:
            self.time_mlp_ins = [
                nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                for _ in range(self.num_action_experts)
            ]
            self.time_mlp_outs = [
                nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                for _ in range(self.num_action_experts)
            ]
        else:
            # Pi0 (non-pi05) shares state_proj across experts and uses
            # action_time_mlp instead of time_mlp_*. We keep parity with
            # single-expert Pi0 for the non-pi05 path to avoid surprise.
            self.state_projs = [
                nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
                for _ in range(self.num_action_experts)
            ]
            self.action_time_mlp_ins = [
                nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
                for _ in range(self.num_action_experts)
            ]
            self.action_time_mlp_outs = [
                nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
                for _ in range(self.num_action_experts)
            ]

        # Correlated-noise Cholesky factors L_eid (one per expert).
        # Loaded lazily from `config.correlation_matrices_path`. When the path
        # is None or correlation_beta == 1.0, this stays None and noise is
        # sampled as plain N(0, I).
        self.correlation_chol = None
        if config.correlation_beta < 1.0 and config.correlation_matrices_path is not None:
            chol_stack = self._load_correlation_chols(
                Path(config.correlation_matrices_path),
                beta=float(config.correlation_beta),
                action_horizon=int(config.action_horizon),
                action_dim=int(config.action_dim),
            )
            if chol_stack is not None:
                # Frozen variable (no gradient), stored on the same device the
                # model lives on. shape: (N, flat_dim, flat_dim).
                self.correlation_chol = nnx.Variable(jnp.asarray(chol_stack, dtype=jnp.float32))

        self.deterministic = True

    # ------------------------------------------------------------------ prefix

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Same as Pi0.embed_prefix — shared backbone path."""
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # ------------------------------------------------------------------ per-expert suffix

    @at.typecheck
    def embed_suffix_for_expert(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        expert_id: int,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        """Build suffix (action+timestep) tokens using ``expert_id``'s IO layers."""
        input_mask = []
        ar_mask = []
        tokens = []

        if not self.pi05:
            # add a single state token (per-expert state_proj)
            state_token = self.state_projs[expert_id](obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_in_proj = self.action_in_projs[expert_id]
        action_tokens = action_in_proj(noisy_actions)
        time_emb = posemb_sincos(timestep, action_in_proj.out_features, min_period=4e-3, max_period=4.0)

        if self.pi05:
            time_emb = self.time_mlp_ins[expert_id](time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_outs[expert_id](time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_ins[expert_id](action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_outs[expert_id](action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------ per-expert loss

    def _sample_noise(
        self,
        rng: at.KeyArrayLike,
        action_shape: tuple[int, ...],
        expert_id: int,
    ) -> at.Float[at.Array, "*b ah ad"]:
        """Sample (B, action_horizon, action_dim) noise.

        If ``self.correlation_chol`` is set, draw noise as
        ``L_eid @ standard_normal_flat`` and reshape; otherwise plain N(0, I).
        """
        z = jax.random.normal(rng, action_shape)
        if self.correlation_chol is None:
            return z
        # action_shape = (B, ah, ad); flatten last two dims.
        b, ah, ad = action_shape
        flat = ah * ad
        z_flat = z.reshape(b, flat)                                          # (B, flat)
        L = self.correlation_chol.value[expert_id]                           # (flat, flat)
        z_correlated = jnp.einsum("ij,bj->bi", L, z_flat)                    # (B, flat)
        return z_correlated.reshape(b, ah, ad)

    def compute_loss_for_expert(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        expert_id: int,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """Forward pass + flow-matching MSE loss for a single expert subset.

        Multi-step flow matching: when ``self.num_flow_samples > 1`` the
        prefix VLM forward runs once and the action expert is unrolled N
        times over fresh (noise, time) tuples, reusing the prefix kv-cache.
        Per-sample MSEs are averaged (variance reduction ∝ 1/√N).
        """
        preprocess_rng, vlm_rng, flow_master_rng = jax.random.split(rng, 3)
        flow_rngs = jax.random.split(flow_master_rng, self.num_flow_samples)  # (N, 2)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # ---- prefix VLM forward (once per call, reused across noise samples) ----
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        prefix_only_tokens: list[jnp.ndarray | None] = [prefix_tokens] + [None] * self.num_action_experts
        outs_prefix, kv_cache = self.PaliGemma.llm(
            prefix_only_tokens,
            mask=prefix_attn_mask,
            positions=prefix_positions,
            adarms_cond=[None] + [None] * self.num_action_experts,
        )
        prefix_out = outs_prefix[0]                                          # (b, prefix_len, paligemma_width)

        def _one_sample_loss(sample_rng):
            noise_rng, time_rng = jax.random.split(sample_rng)
            noise = self._sample_noise(noise_rng, actions.shape, expert_id)
            batch_shape = actions.shape[:-2]
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_for_expert(
                observation, x_t, time, expert_id
            )

            # Build attention mask for suffix that can attend to prefix + own suffix.
            suffix_attn_self = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_for_suffix, suffix_attn_self], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            )

            tokens_list: list[jnp.ndarray | None] = [None] + [None] * self.num_action_experts
            tokens_list[1 + expert_id] = suffix_tokens
            adarms_list: list[jnp.ndarray | None] = [None] + [None] * self.num_action_experts
            adarms_list[1 + expert_id] = adarms_cond

            outs_suffix, _ = self.PaliGemma.llm(
                tokens_list,
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=adarms_list,
            )
            expert_out = outs_suffix[1 + expert_id]
            v_t = self.action_out_projs[expert_id](expert_out[:, -self.action_horizon:])
            return jnp.mean(jnp.square(v_t - u_t), axis=-1)                  # (b, ah)

        # Single-sample fast path (avoids unnecessary list ops + gives identical
        # gradient graph to the legacy pre-multi-step implementation).
        if self.num_flow_samples == 1:
            flow_loss = _one_sample_loss(flow_rngs[0])
        else:
            # lax.scan compiles `_one_sample_loss` once and iterates N times
            # at runtime (vs. statically unrolling N copies into the graph,
            # which makes XLA compile time blow up with N).
            #
            # jax.checkpoint wraps the body so that inside-the-scan activations
            # are *recomputed* during backward instead of stored. Without this,
            # scan saves carry+activations from all N iterations for autodiff,
            # which OOMs at batch=24/gpu × N=15. Recomputation costs ~1 extra
            # forward pass per backward but keeps activation memory at ~1×.
            ckpt_loss = jax.checkpoint(_one_sample_loss)
            def _scan_body(loss_acc, sample_rng):
                # Cast to fp32 so scan carry dtype is stable regardless of
                # whether the model runs in bf16 or fp32.
                return loss_acc + ckpt_loss(sample_rng).astype(jnp.float32), None
            init = jnp.zeros(actions.shape[:-1], dtype=jnp.float32)          # (b, ah)
            total, _ = jax.lax.scan(_scan_body, init, flow_rngs)
            flow_loss = total / self.num_flow_samples                         # (b, ah)

        return flow_loss

    # ------------------------------------------------------------------ batched loss

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        per_expert_counts: tuple[int, ...] | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        """Stratified-batch dispatch: per-expert forwards with VARIABLE slot
        sizes from ``per_expert_counts``.

        The upstream stratified sampler arranges batch samples in contiguous
        per-expert slots of variable size (proportional to head pool size by
        default). Each slot[i] is forwarded through expert ``i`` only.

        ``per_expert_counts`` must be a Python tuple of ints (static under
        JIT — recompiles on change), enabling per-batch slot sizes like
        ``(42, 24, 16, 9, 4, 1)`` for batch=96. Falls back to equal split
        ``B // N`` when not provided (legacy uniform mode).
        """
        skill_ids = observation.skill_canonical_ids
        if skill_ids is None:
            raise ValueError(
                "Pi0SixHead.compute_loss requires `observation.skill_canonical_ids`. "
                "Either feed the dataset's `skill_canonical_id` field through the "
                "data pipeline, or use Pi0 (single-expert) instead."
            )

        B = actions.shape[0]
        N = self.num_action_experts

        # Resolve per-expert counts (Python static).
        if per_expert_counts is None:
            # Legacy uniform mode: equal split.
            if B % N != 0:
                raise ValueError(
                    f"Batch size {B} not divisible by num_action_experts {N} and "
                    f"observation.per_expert_counts not provided; "
                    f"the upstream sampler must produce stratified batches."
                )
            per_expert_counts = (B // N,) * N
        else:
            per_expert_counts = tuple(int(c) for c in per_expert_counts)
            if len(per_expert_counts) != N:
                raise ValueError(
                    f"per_expert_counts length {len(per_expert_counts)} != num_experts {N}"
                )
            if sum(per_expert_counts) != B:
                raise ValueError(
                    f"sum(per_expert_counts)={sum(per_expert_counts)} != batch_size {B}"
                )

        # Fast path: single-head batch (used by per-head sequential training,
        # e.g., v11_e1 stage-2 specialization). When exactly one slot is
        # non-zero, skip the per-expert loop + slicing/concat overhead and
        # dispatch the full batch directly to that expert.
        nonzero_eids = [i for i, c in enumerate(per_expert_counts) if c > 0]
        if len(nonzero_eids) == 1:
            return self.compute_loss_for_expert(
                rng, observation, actions, nonzero_eids[0], train=train
            )

        # Cumulative offsets for slot starts.
        offsets = [0]
        for c in per_expert_counts:
            offsets.append(offsets[-1] + c)

        # Per-expert RNG.
        rngs = jax.random.split(rng, N)

        per_expert_losses: list[jnp.ndarray] = []
        for eid in range(N):
            size = per_expert_counts[eid]
            if size == 0:
                # Skip empty slot entirely — no slice, no expert forward.
                continue
            sub_rng = rngs[eid]
            start = offsets[eid]
            sub_obs = _slice_observation(observation, start, size)
            sub_actions = jax.lax.dynamic_slice_in_dim(actions, start, size, axis=0)
            loss_eid = self.compute_loss_for_expert(sub_rng, sub_obs, sub_actions, eid, train=train)
            per_expert_losses.append(loss_eid)

        flow_loss = jnp.concatenate(per_expert_losses, axis=0)        # (B, ah)

        # Same shape as input — train.py mean-reduces this for the scalar loss.
        return flow_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        expert_id: int | None = None,
        prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
        prev_actions_mask: at.Bool[at.Array, "ah"] | None = None,  # noqa: F821 -- jaxtyping dimension
        inpaint_until_time: float = 0.3,
    ) -> _model.Actions:
        """Sample actions using the expert specified by ``expert_id``.

        ``expert_id`` is meant to be a Python int passed from the host so it can
        be a JIT static argument (one compile per expert, then cached). When
        called eagerly (no jit), pass nothing and we'll fall back to reading
        ``observation.skill_canonical_ids[0]`` Python-side.

        Rolling chunk inpainting: when ``prev_actions`` and
        ``prev_actions_mask`` are provided, masked positions are hard-anchored
        onto the OT path ``x_t = t·noise + (1-t)·prev_actions`` while
        ``time > inpaint_until_time`` (i.e. roughly the first 1 -
        inpaint_until_time fraction of denoise steps). Late steps run free so
        the model can resolve any residual mismatch with the new context.
        """
        if expert_id is None:
            skill_ids = observation.skill_canonical_ids
            if skill_ids is None:
                raise ValueError(
                    "Pi0SixHead.sample_actions requires either expert_id (kwarg) or "
                    "observation.skill_canonical_ids (eager-only fallback)"
                )
            expert_id = int(skill_ids[0])
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        inpaint_active = prev_actions is not None and prev_actions_mask is not None
        if inpaint_active:
            # OT-path target uses the same noise that initializes the denoise
            # loop (matches b1k champion: ``fixed_z_O = noise[O_indices]``).
            # At t=1 the target equals the loop's starting state at masked
            # positions, so the inpaint constraint sits *on* the trajectory
            # the model is descending — not on a separately sampled path.
            mask_b = prev_actions_mask.astype(jnp.bool_).reshape(1, -1, 1)
            prev_actions_cast = prev_actions.astype(noise.dtype)
            inpaint_until = jnp.asarray(inpaint_until_time, dtype=noise.dtype)

        # Fill KV cache with prefix forward.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # Pass [prefix, None, ..., None] — only PaliGemma runs to fill cache.
        prefix_tokens_list: list[jnp.ndarray | None] = [prefix_tokens] + [None] * self.num_action_experts
        _, kv_cache = self.PaliGemma.llm(prefix_tokens_list, mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix_for_expert(
                observation, x_t, jnp.broadcast_to(time, batch_size), expert_id
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_ = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_, suffix_attn_mask], axis=-1)
            positions_ = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            tokens_list: list[jnp.ndarray | None] = [None] + [None] * self.num_action_experts
            tokens_list[1 + expert_id] = suffix_tokens
            adarms_list: list[jnp.ndarray | None] = [None] + [None] * self.num_action_experts
            adarms_list[1 + expert_id] = adarms_cond

            outs, _ = self.PaliGemma.llm(
                tokens_list,
                mask=full_attn_mask,
                positions=positions_,
                kv_cache=kv_cache,
                adarms_cond=adarms_list,
            )
            v_t = self.action_out_projs[expert_id](outs[1 + expert_id][:, -self.action_horizon :])
            x_next = x_t + dt * v_t
            new_time = time + dt
            if inpaint_active:
                target = new_time * noise + (1.0 - new_time) * prev_actions_cast
                do_inpaint = new_time > inpaint_until
                x_next = jnp.where(mask_b & do_inpaint, target, x_next)
            return x_next, new_time

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0


    # ------------------------------------------------------------------ helpers (correlated noise)

    @staticmethod
    def _load_correlation_chols(
        per_expert_dir: Path,
        *,
        beta: float,
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray | None:
        """Load 6 per-expert correlation matrices and return their Cholesky factors.

        Returns
        -------
        L_stack : (num_experts, flat_dim, flat_dim) np.ndarray
            Each L_stack[i] is ``chol(β·I + (1-β)·Σ_i)`` of the i-th expert's
            normalized action correlation matrix. Returns ``None`` if any
            expert directory is missing or has no correlation matrix saved
            (in which case the model falls back to plain N(0, I) sampling).
        """
        from openpi.training.skill_segment_dataset import CANONICAL_HEADS

        flat_dim = action_horizon * action_dim
        chols: list[np.ndarray] = []
        for name in CANONICAL_HEADS:
            stats_dir = per_expert_dir / name
            try:
                stats = _normalize.load(stats_dir)
            except (FileNotFoundError, OSError):
                logger.warning("correlated noise: missing per-expert stats at %s — disabling", stats_dir)
                return None
            actions_stats = stats.get("actions")
            corr = getattr(actions_stats, "correlation_matrix", None) if actions_stats is not None else None
            if corr is None:
                logger.warning("correlated noise: stats at %s have no correlation_matrix — disabling", stats_dir)
                return None
            corr = np.asarray(corr, dtype=np.float64)
            if corr.shape != (flat_dim, flat_dim):
                # Saved correlation may be on the un-padded action_dim (e.g.
                # 23 instead of 32). Pad with identity on the missing dims so
                # the noise sampler sees a (H*32) × (H*32) matrix.
                expected_raw = corr.shape[0]
                if expected_raw % action_horizon != 0:
                    logger.warning(
                        "correlated noise: corr shape %s incompatible with horizon %d — disabling",
                        corr.shape, action_horizon,
                    )
                    return None
                raw_dim = expected_raw // action_horizon
                if raw_dim > action_dim:
                    logger.warning(
                        "correlated noise: corr action_dim=%d > config action_dim=%d — disabling",
                        raw_dim, action_dim,
                    )
                    return None
                # Embed the smaller correlation matrix in a flat_dim×flat_dim
                # block-diagonal structure: real corr on the first raw_dim
                # action dims per timestep, identity on the padding dims.
                corr_padded = np.eye(flat_dim, dtype=np.float64)
                for t_q in range(action_horizon):
                    for t_k in range(action_horizon):
                        block_q = t_q * action_dim
                        block_k = t_k * action_dim
                        src_q = t_q * raw_dim
                        src_k = t_k * raw_dim
                        corr_padded[block_q:block_q + raw_dim, block_k:block_k + raw_dim] = (
                            corr[src_q:src_q + raw_dim, src_k:src_k + raw_dim]
                        )
                corr = corr_padded
            # Regularize: β·I + (1-β)·Σ; small jitter on diag for numerical stability.
            sigma_reg = beta * np.eye(flat_dim) + (1.0 - beta) * corr
            sigma_reg += 1e-6 * np.eye(flat_dim)
            try:
                L = np.linalg.cholesky(sigma_reg)
            except np.linalg.LinAlgError as e:
                logger.warning("correlated noise: Cholesky failed for %s (%s) — disabling", name, e)
                return None
            chols.append(L.astype(np.float32))
        return np.stack(chols, axis=0)


def _slice_observation(obs: _model.Observation, start: int, length: int) -> _model.Observation:
    """Static-shape slice along the batch axis (works under jit via dynamic_slice_in_dim)."""

    def _slice(x):
        return jax.lax.dynamic_slice_in_dim(x, start, length, axis=0)

    return _model.Observation(
        images={k: _slice(v) for k, v in obs.images.items()},
        image_masks={k: _slice(v) for k, v in obs.image_masks.items()},
        state=_slice(obs.state),
        tokenized_prompt=_slice(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None,
        tokenized_prompt_mask=_slice(obs.tokenized_prompt_mask) if obs.tokenized_prompt_mask is not None else None,
        token_ar_mask=_slice(obs.token_ar_mask) if obs.token_ar_mask is not None else None,
        token_loss_mask=_slice(obs.token_loss_mask) if obs.token_loss_mask is not None else None,
        skill_canonical_ids=_slice(obs.skill_canonical_ids) if obs.skill_canonical_ids is not None else None,
    )
