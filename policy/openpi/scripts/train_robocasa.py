"""Six-GPU v3 control: frozen shared VLM and six trainable action experts."""
import pandas as _pandas_preload

import argparse
import dataclasses
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import train as base_train
from train_robocasa_joint import (
    AssetProvider, OUTPUT, head_for_path, recipe as joint_recipe, validate_assets,
)
from openpi.models import model as model_lib
from openpi.training import checkpoints, sharding
from openpi.training.robocasa_data import BUDGETS, ROOT, SKILLS, counts_at, torch_loader


def frozen_vlm(path, value):
    del value
    return head_for_path(path) == -1


def recipe(exp_name='frozen_vlm', resume=False):
    return dataclasses.replace(joint_recipe(exp_name, resume),
        name='pi05_robocasa_frozen_vlm_v3', freeze_filter=frozen_vlm,
        batch_size=48, fsdp_devices=2)


def padded_counts(counts):
    return tuple(12 if count else 0 for count in counts)


def pad_batch(raw, counts):
    indices = []
    offset = 0
    for count in counts:
        if count:
            assert count == 8
            indices.extend(range(offset, offset + count))
            indices.extend([offset] * 4)
            offset += count
    assert offset == raw['actions'].shape[0]
    return jax.tree.map(lambda value: value[np.asarray(indices)], raw)


def valid_mask(counts):
    return jnp.asarray([i < 8 for count in counts if count for i in range(12)])


def apply_grads(cfg, counts, state, grads):
    trainable = state.params.filter(cfg.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, trainable)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, optax.apply_updates(trainable, updates))
    params = nnx.state(model)

    def ema(path, old, new):
        head = head_for_path(path)
        if head == -1 or counts[head] == 0:
            return old
        return cfg.ema_decay * old + (1 - cfg.ema_decay) * new

    ema_params = jax.tree_util.tree_map_with_path(ema, state.ema_params, params)
    return dataclasses.replace(state, step=state.step + 1, params=params,
                               opt_state=opt_state, ema_params=ema_params)


def step_fn(cfg, counts, rng, state, batch):
    rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    model = nnx.merge(state.model_def, state.params)
    model.train()
    mask = valid_mask(counts)

    def loss_fn(model):
        losses = model.compute_loss(rng, observation, actions, train=True,
                                    per_expert_counts=padded_counts(counts))
        per_sample = jnp.mean(losses, axis=tuple(range(1, losses.ndim)))
        # Padding never changes real sample counts or the global loss denominator.
        return jnp.sum(jnp.where(mask, per_sample, 0)) / sum(counts), per_sample

    (loss, per_sample), grads = nnx.value_and_grad(
        loss_fn, argnums=nnx.DiffState(0, cfg.trainable_filter), has_aux=True)(model)
    grouped = {i: [] for i in range(6)}
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        head = head_for_path(path)
        if head < 0:
            raise RuntimeError('Frozen VLM unexpectedly present in gradient tree')
        grouped[head].append(leaf)
    metrics = {'loss': loss, 'grad_norm': optax.global_norm(grads), 'grad/vlm': jnp.asarray(0.)}
    for index, name in enumerate(SKILLS):
        selected = mask & (observation.skill_canonical_ids == index)
        metrics['count/skill_' + name] = jnp.sum(selected)
        metrics['loss/skill_' + name] = (
            jnp.sum(jnp.where(selected, per_sample, 0)) / max(counts[index], 1))
        metrics['grad/' + name] = optax.global_norm(grouped[index])
    return apply_grads(cfg, counts, state, grads), metrics


def vlm_digest(params, cfg):
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(params.filter(cfg.freeze_filter))[0]:
        array = np.asarray(jax.device_get(leaf))
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_frozen(state, cfg, expected):
    actual = vlm_digest(state.params, cfg)
    ema = vlm_digest(state.ema_params, cfg)
    if actual != expected or ema != expected:
        raise RuntimeError('Frozen VLM or its EMA changed')
    logging.info('FROZEN VLM VERIFIED step=%d sha256=%s', int(state.step), actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exp-name', default='frozen_vlm')
    parser.add_argument('--stop-after', type=int, default=45000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    base_train.init_logging()
    cfg = recipe(args.exp_name, args.resume)
    assert 2 <= args.stop_after <= 45000
    visible = [int(value) for value in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    assert len(visible) == 6 and len(set(visible)) == 6
    import pynvml as nv
    nv.nvmlInit()
    try:
        for index in visible:
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            others = [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid != os.getpid()]
            if others or nv.nvmlDeviceGetMemoryInfo(handle).free < 70 * 2**30:
                raise RuntimeError(f'GPU {index} unavailable: {others}')
    finally:
        nv.nvmlShutdown()
    assert jax.device_count() == 6
    if shutil.disk_usage(OUTPUT).free < (65 if args.resume else 120) * 2**30:
        raise RuntimeError('Insufficient checkpoint replacement space')
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,
        keep_period=None, overwrite=False, resume=args.resume, replica_parallel=False)
    assets = AssetProvider(cfg)
    rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    state, state_sharding = base_train.init_train_state(cfg, init_rng, mesh, resume=resuming)
    if resuming:
        validate_assets(cfg.checkpoint_dir / str(manager.latest_step()))
        state = checkpoints.restore_state(manager, state, assets)
        # Orbax's legacy restore retains the saved layout; a new mesh needs
        # explicit resharding before the jitted train step accepts the state.
        state = jax.device_put(state, state_sharding)
    jax.block_until_ready(state)
    start = int(state.step)
    manifest = cfg.checkpoint_dir / 'frozen_recipe.json'
    if args.resume:
        saved = json.loads(manifest.read_text())
        expected = saved['frozen_vlm_sha256']
        assert start >= 2 and saved['normalization'] == 'per_skill_mean_std'
        assert saved['skills'] == list(SKILLS) and saved['budgets'] == list(BUDGETS)
        verify_frozen(state, cfg, expected)
        saved.setdefault('initial_fsdp_devices', saved['fsdp_devices'])
        saved['fsdp_devices'] = cfg.fsdp_devices
        saved['data_parallel_groups'] = 6 // cfg.fsdp_devices
        saved.setdefault('resume_mesh_history', []).append({
            'step': start, 'fsdp_devices': cfg.fsdp_devices,
            'data_parallel_groups': 6 // cfg.fsdp_devices, 'timestamp': time.time()})
        manifest.write_text(json.dumps(saved, indent=2) + '\n')
    else:
        expected = vlm_digest(state.params, cfg)
        groups = {i: 0 for i in range(-1, 6)}
        for path, leaf in jax.tree_util.tree_flatten_with_path(state.params)[0]:
            groups[head_for_path(path)] += leaf.size
        manifest.write_text(json.dumps({
            'skills': SKILLS, 'budgets': BUDGETS, 'initial_params': str(cfg.weight_loader.params_path),
            'frozen_vlm_sha256': expected, 'frozen_vlm_dtype': 'bfloat16',
            'frozen_vlm_parameters': groups[-1], 'trainable_parameters': sum(groups[i] for i in range(6)),
            'gpus': visible, 'fsdp_devices': cfg.fsdp_devices,
            'data_parallel_groups': 6 // cfg.fsdp_devices, 'per_active_head_real_batch': 8,
            'per_active_head_padded_batch': 12, 'padding_in_loss': False,
            'normalization': 'per_skill_mean_std', 'ema_decay': cfg.ema_decay,
            'seed': cfg.seed, 'model': dataclasses.asdict(cfg.model),
            'total_sample_exposures': sum(BUDGETS) * 8,
            'note': 'Frozen VLM control from human300; not a resume of the joint-trained VLM.',
        }, indent=2) + '\n')
        verify_frozen(state, cfg, expected)
    logging.info('STATE READY step=%d devices=6 frozen_vlm=True checkpoint=%s', start, cfg.checkpoint_dir)
    loader = torch_loader(start=start, stop=args.stop_after, workers=2)
    compiled = {}
    try:
        for step, raw in enumerate(loader, start=start):
            counts = counts_at(step)
            np.testing.assert_array_equal(raw['skill_canonical_ids'], np.repeat(np.arange(6), counts))
            assert raw['actions'].shape == (sum(counts), 50, 32)
            assert np.isfinite(raw['actions']).all() and np.isfinite(raw['state']).all()
            raw = pad_batch(raw, counts)
            observation = model_lib.Observation.from_dict(raw)
            batch = jax.tree.map(lambda x: jax.device_put(x, data_sharding), (observation, raw['actions']))
            if counts not in compiled:
                compiled[counts] = jax.jit(functools.partial(step_fn, cfg, counts),
                    in_shardings=(replicated, state_sharding, data_sharding),
                    out_shardings=(state_sharding, replicated), donate_argnums=(1,))
                logging.info('Compiling phase step=%d real_counts=%s padded_counts=%s',
                             step, counts, padded_counts(counts))
            before = time.monotonic()
            with sharding.set_mesh(mesh):
                state, info = compiled[counts](rng, state, batch)
            metrics = {key: float(value) for key, value in jax.device_get(info).items()}
            assert all(np.isfinite(value) for value in metrics.values()), metrics
            assert metrics['grad/vlm'] == 0
            if step < 2:
                assert all(metrics['grad/' + name] > 0 for name in SKILLS)
            metrics.update(step=step + 1, timestamp=time.time(), seconds=time.monotonic() - before,
                           per_expert_counts=counts, padded_per_expert_counts=padded_counts(counts))
            with (cfg.checkpoint_dir / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metrics) + '\n')
            if step < 2 or (step + 1) % 10 == 0:
                logging.info('TRAIN step=%d loss=%.6f grad=%.6f vlm_grad=0 seconds=%.2f',
                             step + 1, metrics['loss'], metrics['grad_norm'], metrics['seconds'])
            if (step + 1) % 5000 == 0 or step + 1 == args.stop_after:
                verify_frozen(state, cfg, expected)
                if shutil.disk_usage(OUTPUT).free < 65 * 2**30:
                    raise RuntimeError('Insufficient space for atomic checkpoint replacement')
                checkpoints.save_state(manager, state, assets, step + 1)
                manager.wait_until_finished()
                validate_assets(cfg.checkpoint_dir / str(step + 1))
                logging.info('CHECKPOINT VERIFIED step=%d', step + 1)
        if args.stop_after == 2:
            (cfg.checkpoint_dir / 'smoke_passed.json').write_text(json.dumps({
                'steps': 2, 'frozen_vlm_unchanged': True, 'frozen_vlm_ema_unchanged': True,
                'all_expert_gradients_nonzero': True, 'checkpoint_assets_verified': True,
                'timestamp': time.time()}) + '\n')
    finally:
        manager.wait_until_finished()
        manager.close()


if __name__ == '__main__':
    main()
