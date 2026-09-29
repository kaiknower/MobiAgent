# NOTE: pandas must be imported BEFORE torch / openpi.models.model on this
# machine. torch (loaded via openpi.models.model) brings in a libstdc++/libgomp
# whose dlopen ordering corrupts pandas's later native-extension load (segfault
# in pandas/_libs/pandas_parser.so). Reordering avoids the crash; functionally
# identical otherwise (pandas is imported eventually via the data loader).
import pandas as _pandas_preload  # noqa: F401  -- order-of-import fix

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    wandb_id_file = ckpt_dir / "wandb_id.txt"
    if resuming and wandb_id_file.exists():
        run_id = wandb_id_file.read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # Either fresh start, or resuming from ckpt copied from another exp_name
        # (no wandb_id.txt yet). Start a new wandb run and persist its id.
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        wandb_id_file.write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _compute_grads_step(
    config: _config.TrainConfig,
    per_expert_counts: tuple[int, ...] | None,
    train_rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    """Compute gradients + per-microstep diagnostics. No optimizer step.
    train_rng is expected to be ALREADY FOLDED by the caller (per microstep / per optim step).
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        kwargs = {} if per_expert_counts is None else {"per_expert_counts": per_expert_counts}
        chunked_loss = model.compute_loss(rng, observation, actions, train=True, **kwargs)
        per_sample_loss = jnp.mean(chunked_loss, axis=tuple(range(1, chunked_loss.ndim)))
        sample_mean_loss = jnp.mean(per_sample_loss)

        loss = sample_mean_loss
        per_head_loss_weight = float(getattr(config, "per_head_loss_weight", 0.0))
        skill_ids_for_loss = observation.skill_canonical_ids
        if per_head_loss_weight > 0.0 and skill_ids_for_loss is not None:
            head_means = []
            for k in range(6):
                mask = (skill_ids_for_loss == k).astype(per_sample_loss.dtype)
                n = jnp.sum(mask)
                mean_k = jnp.sum(per_sample_loss * mask) / jnp.maximum(n, 1.0)
                head_means.append(jnp.where(n > 0, mean_k, sample_mean_loss))
            head_macro_loss = jnp.mean(jnp.stack(head_means))
            w = jnp.asarray(per_head_loss_weight, dtype=sample_mean_loss.dtype)
            loss = (1.0 - w) * sample_mean_loss + w * head_macro_loss

        return loss, per_sample_loss

    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_sample_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
    }

    # Per-skill_canonical loss breakdown — diagnostic for multi-task averaging.
    # Order matches openpi.training.skill_segment_dataset.CANONICAL_HEADS:
    # 0=move_to, 1=pick_up_from, 2=place_in, 3=place_on, 4=open, 5=close.
    skill_ids = observation.skill_canonical_ids
    if skill_ids is not None:
        head_names = ("move_to", "pick_up_from", "place_in", "place_on", "open", "close")
        for k, name in enumerate(head_names):
            mask = (skill_ids == k).astype(per_sample_loss.dtype)
            n = jnp.sum(mask)
            mean_k = jnp.sum(per_sample_loss * mask) / jnp.maximum(n, 1.0)
            info[f"loss/skill_{name}"] = jnp.where(n > 0, mean_k, jnp.asarray(jnp.nan, mean_k.dtype))
            info[f"count/skill_{name}"] = n

    return grads, info


def _apply_grads_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    grads,
):
    """Apply (already-averaged) gradients to update state. EMA + param_norm here."""
    model = nnx.merge(state.model_def, state.params)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {"param_norm": optax.global_norm(kernel_params)}
    return new_state, info


def train_step(
    config: _config.TrainConfig,
    per_expert_counts: tuple[int, ...] | None,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Single-microstep path (no grad accumulation). Backward-compat with grad_accum_steps=1."""
    train_rng = jax.random.fold_in(rng, state.step)
    grads, info = _compute_grads_step(config, per_expert_counts, train_rng, state, batch)
    new_state, apply_info = _apply_grads_step(config, state, grads)
    info.update(apply_info)
    return new_state, info


def make_n_microstep_train_step(
    config: _config.TrainConfig,
    per_expert_counts: tuple[int, ...] | None,
    N: int,
):
    """Factory: returns a function that does N microbatches + 1 optim step.
    Uses jax.lax.scan over microbatches (NOT Python for loop) so XLA executes
    iterations sequentially, freeing each microstep's grads/activations before
    the next one. Python unroll caused N× grad memory and OOMed at N=4.
    """
    def n_microstep_step(
        rng: at.KeyArrayLike,
        state: training_utils.TrainState,
        batches_tuple,  # tuple of N (Observation, Actions) pairs
    ):
        # Stack the N batches along axis 0 so scan can iterate over them.
        stacked_batches = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *batches_tuple)

        # Run the first microbatch outside scan to obtain the grads pytree
        # structure (used as scan's initial carry).
        first_batch = jax.tree.map(lambda x: x[0], stacked_batches)
        micro_rng_0 = jax.random.fold_in(rng, state.step * N + 0)
        grads_0, info_0 = _compute_grads_step(
            config, per_expert_counts, micro_rng_0, state, first_batch
        )

        if N > 1:
            # Scan over remaining N-1 microbatches; XLA executes them sequentially
            # so peak grads memory ≈ 1× (one set in flight at any time).
            rest_batches = jax.tree.map(lambda x: x[1:], stacked_batches)

            def scan_body(carry_grads, scan_input):
                batch_i, micro_idx = scan_input
                micro_rng = jax.random.fold_in(rng, state.step * N + micro_idx)
                grads_i, info_i = _compute_grads_step(
                    config, per_expert_counts, micro_rng, state, batch_i
                )
                new_carry = jax.tree.map(jnp.add, carry_grads, grads_i)
                return new_carry, info_i

            scan_idx = jnp.arange(1, N, dtype=jnp.int32)
            sum_grads, infos_rest = jax.lax.scan(scan_body, grads_0, (rest_batches, scan_idx))
        else:
            sum_grads = grads_0
            infos_rest = jax.tree.map(
                lambda x: jnp.empty((0,) + x.shape, x.dtype), info_0
            )

        avg_grads = jax.tree.map(lambda g: g / N, sum_grads)
        new_state, apply_info = _apply_grads_step(config, state, avg_grads)

        # Aggregate per-microstep info: mean for losses/grad_norm; sum for counts.
        # info_0 is per-microstep scalars; infos_rest has leading dim N-1.
        info = {}
        for k in info_0.keys():
            first = jnp.expand_dims(info_0[k], 0)  # shape (1,)
            rest = infos_rest[k]                    # shape (N-1,)
            stacked_v = jnp.concatenate([first, rest], axis=0)  # shape (N,)
            info[k] = jnp.sum(stacked_v) if k.startswith("count/") else jnp.mean(stacked_v)
        info.update(apply_info)
        return new_state, info
    return n_microstep_step


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # If the underlying batch sampler is a stratified sampler with variable
    # per-expert slot sizes, capture them as a Python tuple and thread into
    # train_step. compute_loss uses these as static slice sizes when present.
    per_expert_counts: tuple[int, ...] | None = None
    # Walk through nested ``_data_loader`` wrappers to reach the torch DataLoader,
    # which exposes ``batch_sampler``. Structure: DataLoaderImpl → TorchDataLoader → torch.DataLoader.
    inner_sampler = None
    cursor = data_loader
    for _ in range(5):
        bs = getattr(cursor, "batch_sampler", None)
        if bs is not None and hasattr(bs, "per_expert"):
            inner_sampler = bs
            break
        cursor = getattr(cursor, "_data_loader", None)
        if cursor is None:
            break
    if inner_sampler is not None:
        try:
            counts = inner_sampler.per_expert  # property returning list[int]
            if callable(counts):
                counts = counts()
            per_expert_counts = tuple(int(c) for c in counts)
            logging.info(f"train_step: per_expert_counts (static) = {per_expert_counts}")
        except Exception as e:
            logging.warning(f"Could not extract per_expert_counts from sampler: {e}")
    else:
        logging.warning(
            "train_step: no stratified batch_sampler found; per_expert_counts=None "
            "(compute_loss will fall back to uniform B//N split — wrong for non-uniform batches!)"
        )

    # ── alignment guard: pull one batch from the loader and verify its
    # `skill_canonical_ids` slot k contains exactly k for k in range(N). This
    # is the invariant the 6-head compute_loss relies on; if the upstream
    # sampler / collate ever reorders rows, expert k would receive wrong-head
    # data (the prior "训错数据" failure mode). Cost: ~1 batch (already
    # prefetched anyway). Only runs when stratified counts are present.
    if per_expert_counts is not None:
        try:
            _verify_iter = iter(data_loader)
            _verify_obs, _ = next(_verify_iter)
            del _verify_iter
            _ids = getattr(_verify_obs, "skill_canonical_ids", None)
            if _ids is None:
                logging.warning("alignment guard: skill_canonical_ids missing from observation — skipping check")
            else:
                _ids_np = np.asarray(_ids)
                _off = 0
                for _eid, _c in enumerate(per_expert_counts):
                    if _c == 0:
                        continue
                    _slot = _ids_np[_off:_off + _c]
                    _off += _c
                    if not np.all(_slot == _eid):
                        _bad = np.where(_slot != _eid)[0]
                        raise RuntimeError(
                            f"alignment guard FAILED at expert {_eid}: slot "
                            f"[{_off-_c}:{_off}] contains non-{_eid} ids "
                            f"(first {min(5, len(_bad))} bad positions: {_bad[:5].tolist()}, "
                            f"values: {_slot[_bad[:5]].tolist()}). "
                            f"This is the 'wrong-head data' bug — sampler / collate "
                            f"must emit contiguous-by-expert batches."
                        )
                logging.info(
                    "alignment guard PASSED: per_expert_counts=%s, slot ids "
                    "exactly match expert indices.", per_expert_counts,
                )
        except StopIteration:
            logging.warning("alignment guard: data loader yielded no batches — skipping check")
        except RuntimeError:
            raise
        except Exception as _e:
            logging.warning(f"alignment guard: skipped due to {type(_e).__name__}: {_e}")

    grad_accum_steps = max(1, getattr(config, "grad_accum_steps", 1))
    if grad_accum_steps > 1:
        logging.info(
            f"Gradient accumulation ENABLED: grad_accum_steps={grad_accum_steps}. "
            f"Each optim step processes {grad_accum_steps}× batch_size={config.batch_size} "
            f"= {grad_accum_steps * config.batch_size} samples (effective)."
        )
        # Single jit with N microsteps + apply inside → grads stay FSDP-sharded.
        n_step_fn = make_n_microstep_train_step(config, per_expert_counts, grad_accum_steps)
        ptrain_step = jax.jit(
            n_step_fn,
            in_shardings=(replicated_sharding, train_state_sharding, (data_sharding,) * grad_accum_steps),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config, per_expert_counts),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if grad_accum_steps > 1:
                # Collect N microbatches as a tuple, then call combined jit.
                # `batch` is the first one (already prefetched); fetch the rest.
                batches = (batch,) + tuple(next(data_iter) for _ in range(grad_accum_steps - 1))
                train_state, info = ptrain_step(train_rng, train_state, batches)
            else:
                train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            mem_lines = []
            for d in jax.devices():
                stats = d.memory_stats() or {}
                bytes_in_use = stats.get("bytes_in_use", 0) / (1024**3)
                peak = stats.get("peak_bytes_in_use", 0) / (1024**3)
                mem_lines.append(f"GPU{d.id} {bytes_in_use:.1f}/{peak:.1f}GiB")
            pbar.write(f"Step {step}: {info_str} | mem (cur/peak): {' '.join(mem_lines)}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
