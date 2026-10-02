"""RoboCasa joint training: shared VLM and six action experts (eight GPUs)."""
import pandas as _pandas_preload  # native library import order

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.training import runner as base_train
from openpi.models import model as model_lib
from openpi.training import checkpoints, config, optimizer, sharding, weight_loaders
from openpi.training.robocasa_data import (
    BUDGETS, SKILLS, RoboCasaV3Data, counts_at, model_config, source_recipe, torch_loader,
)

OUTPUT = Path(os.environ.get('MOBIAGENT_CHECKPOINT_DIR', 'checkpoints/robocasa'))


def head_for_path(path):
    parts = [str(getattr(p, 'key', getattr(p, 'idx', getattr(p, 'name', p)))) for p in path]
    for index, part in enumerate(parts):
        if part in ('action_in_projs', 'action_out_projs', 'time_mlp_ins', 'time_mlp_outs'):
            return int(parts[index + 1])
    if 'llm' in parts:
        for part in parts:
            match = re.search(r'_([1-6])$', part)
            if match:
                return int(match.group(1)) - 1
    return -1


@dataclasses.dataclass(frozen=True)
class V3Optimizer:
    def create(self, lr, weight_decay_mask=None):
        assert weight_decay_mask is None

        def schedule(budget):
            decay = optimizer.CosineDecaySchedule(decay_steps=budget).create()
            return lambda step: jnp.where(step < budget, decay(step), 0.0)

        schedules = {-1: lr, **{i: schedule(b) for i, b in enumerate(BUDGETS)}}
        transforms = {str(i): optax.adamw(fn, b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10)
                      for i, fn in schedules.items()}
        labels = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, _: str(head_for_path(path)), params)
        return optax.chain(optax.clip_by_global_norm(1.0), optax.multi_transform(transforms, labels))


@dataclasses.dataclass(frozen=True)
class StrictBaseLoader(weight_loaders.Pi05BaseToSixHeadLoader):
    def load(self, params):
        result = super().load(params)
        missing = [jax.tree_util.keystr(path) for path, value in jax.tree_util.tree_flatten_with_path(result)[0]
                   if isinstance(value, jax.ShapeDtypeStruct)]
        if missing:
            raise ValueError(f'Uninitialized checkpoint leaves: {missing}')
        return result


def recipe(exp_name, resume=False):
    heads = source_recipe(validate_datasets=True)
    return config.TrainConfig(name='pi05_robocasa_shared_vlm_v3', exp_name=exp_name,
        model=model_config(), data=RoboCasaV3Data(), freeze_filter=nnx.Nothing(),
        optimizer=V3Optimizer(), lr_schedule=optimizer.CosineDecaySchedule(decay_steps=45000),
        weight_loader=StrictBaseLoader(heads['close']['base_params']),
        batch_size=48, num_train_steps=45000, num_workers=2, fsdp_devices=8,
        ema_decay=0.99, seed=42, checkpoint_base_dir=str(OUTPUT),
        save_interval=5000, keep_period=None, log_interval=10,
        wandb_enabled=False, resume=resume, overwrite=False)


def step_fn(cfg, counts, rng, state, batch):
    rng = jax.random.fold_in(rng, state.step)
    grads, metrics = base_train._compute_grads_step(cfg, counts, rng, state, batch)
    # Audit all gradient groups; inactive towers must stay exactly unchanged.
    grouped = {i: [] for i in range(-1, 6)}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        grouped[head_for_path(path)].append(leaf)
    for i, values in grouped.items():
        metrics['grad/' + ('vlm' if i == -1 else SKILLS[i])] = optax.global_norm(values)
    updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    def ema(path, old, new):
        head = head_for_path(path)
        if head >= 0 and counts[head] == 0:
            return old
        return cfg.ema_decay * old + (1 - cfg.ema_decay) * new

    ema_params = jax.tree_util.tree_map_with_path(ema, state.ema_params, params)
    state = dataclasses.replace(state, step=state.step + 1, params=params,
                                opt_state=opt_state, ema_params=ema_params)
    return state, metrics


class AssetProvider:
    def __init__(self, cfg):
        self.value = cfg.data.create(cfg.assets_dirs, cfg.model)

    def data_config(self):
        return self.value


def validate_assets(directory):
    names = json.loads((directory / 'assets/expert_names.json').read_text())
    assert tuple(names) == SKILLS
    heads = source_recipe()
    from openpi.shared import normalize
    for name in names:
        expected = normalize.load(Path(heads[name]['norm_path']).parent)
        actual = normalize.load(directory / 'assets/per_expert' / name)
        for key in expected:
            for field in ('mean', 'std', 'q01', 'q99'):
                np.testing.assert_array_equal(getattr(actual[key], field), getattr(expected[key], field))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-name', default='joint')
    parser.add_argument('--stop-after', type=int, default=45000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--recipe', type=Path, help='Generated RoboCasa training recipe')
    parser.add_argument('--check-data', action='store_true', help='Validate one sample per expert and exit before GPU allocation')
    args = parser.parse_args()
    if args.recipe:
        os.environ['MOBIAGENT_ROBOCASA_RECIPE'] = str(args.recipe.expanduser().resolve())
    logging.basicConfig(level=logging.INFO)
    base_train.init_logging()
    cfg = recipe(args.exp_name, args.resume)
    if args.check_data:
        from openpi.training.robocasa_data import V3Dataset
        dataset = V3Dataset()
        for expert, name in enumerate(SKILLS):
            sample = dataset[(0, expert, 0)]
            assert sample['state'].shape == (32,) and sample['actions'].shape == (50, 32)
            assert np.isfinite(sample['state']).all() and np.isfinite(sample['actions']).all()
            logging.info('Validated %s dataset sample and normalization', name)
        return
    if not 2 <= args.stop_after <= 45000:
        raise ValueError('stop-after must be between 2 and 45000')
    import pynvml as nv
    nv.nvmlInit()
    try:
        for index in range(8):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            others = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid != os.getpid()]
            if others or nv.nvmlDeviceGetMemoryInfo(handle).free < 70 * 2**30:
                raise RuntimeError(f'GPU {index} not available: {others}')
    finally:
        nv.nvmlShutdown()
    assert jax.device_count() == 8, jax.devices()
    # Keep space for both a new checkpoint and the existing latest one.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    required = 100 if args.resume else 180
    if shutil.disk_usage(OUTPUT).free < required * 2**30:
        raise RuntimeError(f'Need at least {required} GiB free on checkpoint filesystem')
    mesh = sharding.make_mesh(8)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,
        keep_period=None, overwrite=False, resume=args.resume)
    assets = AssetProvider(cfg)
    rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    state, state_sharding = base_train.init_train_state(cfg, init_rng, mesh, resume=resuming)
    if resuming:
        validate_assets(cfg.checkpoint_dir / str(manager.latest_step()))
        state = checkpoints.restore_state(manager, state, assets)
    jax.block_until_ready(state)
    start = int(state.step)
    if args.resume and start < 2:
        raise RuntimeError('Formal resume requires a completed two-step checkpoint')
    logging.info('STATE READY: step=%d devices=%d checkpoint=%s', start, jax.device_count(), cfg.checkpoint_dir)
    cfg.checkpoint_dir.joinpath('joint_recipe.json').write_text(json.dumps({
        'skills': SKILLS, 'budgets': BUDGETS, 'per_active_head_batch': 8,
        'total_sample_exposures': sum(BUDGETS) * 8, 'shared_vlm_steps': 45000,
        'shared_vlm_lr_decay_steps': 45000, 'expert_lr_decay_steps': BUDGETS,
        'model': dataclasses.asdict(cfg.model), 'ema_decay': cfg.ema_decay,
        'optimizer': 'AdamW b1=.9 b2=.95 eps=1e-8 wd=1e-10 global_clip=1',
        'normalization': 'per_skill_mean_std',
    }, indent=2) + '\n')
    loader = torch_loader(start=start, stop=args.stop_after, workers=2)
    compiled = {}
    metrics_path = cfg.checkpoint_dir / 'metrics.jsonl'
    try:
        for step, raw in enumerate(loader, start=start):
            counts = counts_at(step)
            expected_ids = np.repeat(np.arange(6, dtype=np.int32), counts)
            np.testing.assert_array_equal(raw['skill_canonical_ids'], expected_ids)
            assert raw['actions'].shape == (sum(counts), 50, 32)
            assert np.isfinite(raw['actions']).all() and np.isfinite(raw['state']).all()
            observation = model_lib.Observation.from_dict(raw)
            batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), (observation, raw['actions']))
            if counts not in compiled:
                compiled[counts] = jax.jit(functools.partial(step_fn, cfg, counts),
                    in_shardings=(replicated, state_sharding, data_sharding),
                    out_shardings=(state_sharding, replicated), donate_argnums=(1,))
                logging.info('Compiling phase: step=%d per_expert_counts=%s', step, counts)
            before = time.monotonic()
            with sharding.set_mesh(mesh):
                state, info = compiled[counts](rng, state, batch)
            info = jax.device_get(info)
            metrics = {key: float(value) for key, value in info.items()}
            for key, value in metrics.items():
                if key.startswith('loss/skill_') and counts[SKILLS.index(key.removeprefix('loss/skill_'))] == 0:
                    continue
                if not np.isfinite(value):
                    raise FloatingPointError((key, value))
            if step < 2:
                assert all(metrics['grad/' + name] > 0 for name in ('vlm', *SKILLS))
            metrics.update(step=step + 1, seconds=time.monotonic() - before,
                           timestamp=time.time(), per_expert_counts=counts)
            with metrics_path.open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            if step < 2 or (step + 1) % 10 == 0:
                logging.info('TRAIN step=%d loss=%.6f grad=%.6f seconds=%.2f',
                             step + 1, metrics['loss'], metrics['grad_norm'], metrics['seconds'])
            if (step + 1) % 5000 == 0 or step + 1 == args.stop_after:
                if shutil.disk_usage(OUTPUT).free < 92 * 2**30:
                    raise RuntimeError('Insufficient free space for atomic checkpoint replacement')
                checkpoints.save_state(manager, state, assets, step + 1)
                manager.wait_until_finished()
                validate_assets(cfg.checkpoint_dir / str(step + 1))
                logging.info('CHECKPOINT VERIFIED step=%d', step + 1)
        if args.stop_after == 2:
            (cfg.checkpoint_dir / 'smoke_passed.json').write_text(json.dumps(
                {'steps': 2, 'gradients': 'shared VLM and all six experts nonzero and finite',
                 'per_expert_assets_verified': True, 'timestamp': time.time()}) + '\n')
    finally:
        manager.wait_until_finished()
        manager.close()


if __name__ == '__main__':
    main()
