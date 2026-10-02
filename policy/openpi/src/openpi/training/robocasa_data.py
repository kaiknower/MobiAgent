"""RoboCasa data contract for a shared VLM and six independent experts."""
from __future__ import annotations

import dataclasses
import os
import json
from pathlib import Path

import numpy as np
import torch

from openpi import transforms
from openpi.models.pi0_six_head_config import Pi0SixHeadConfig
from openpi.shared import normalize
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from openpi.training.transforms_normalize import PerExpertNormalize
from openpi.training.robocasa_lerobot import ACTION_KEYS, CAMERAS, STATE_KEYS, RoboCasaLeRobotDataset

SKILLS = ('close', 'open', 'switch', 'manipulate', 'navigate', 'pnp')
BUDGETS = (30000, 30000, 35000, 30000, 25000, 45000)
ROOT = Path(__file__).resolve().parents[4]
RECIPE = Path(os.environ.get('MOBIAGENT_ROBOCASA_RECIPE', 'configs/robocasa/data.json'))


def counts_at(step):
    return tuple(8 if step < budget else 0 for budget in BUDGETS)


def source_recipe(*, validate_datasets=False):
    path = Path(os.environ.get('MOBIAGENT_ROBOCASA_RECIPE', str(RECIPE))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Training recipe not found: {path}. Run scripts/robocasa/prepare_data.py first.')
    report = json.loads(path.read_text())
    heads = report['skills']
    if set(heads) != set(SKILLS):
        raise ValueError(f'Recipe must define these skills: {SKILLS}')
    for name in SKILLS:
        for key in ('norm_path', 'base_params', 'data_dirs', 'batch_size', 'action_horizon', 'use_quantile_norm'):
            if key not in heads[name]:
                raise ValueError(f'Missing {name}.{key} in {path}')
        if heads[name]['batch_size'] != 8 or heads[name]['action_horizon'] != 50:
            raise ValueError(f'{name}: this trainer requires batch_size=8 and action_horizon=50')
        if validate_datasets and not heads[name]['data_dirs']:
            raise ValueError(f'No datasets configured for {name}')
        if not Path(heads[name]['norm_path']).is_file():
            raise FileNotFoundError(f'Missing {name} normalization: {heads[name]["norm_path"]}')
        for entry in heads[name]['data_dirs'] if validate_datasets else ():
            if not (Path(entry['path']) / 'meta/info.json').is_file():
                raise FileNotFoundError(f'Missing {name} dataset: {entry["path"]}')
    if len({heads[name]['base_params'] for name in SKILLS}) != 1:
        raise ValueError('All six experts must use the same base checkpoint')
    if len({heads[name]['use_quantile_norm'] for name in SKILLS}) != 1:
        raise ValueError('All six experts must use the same normalization mode')
    return heads


def model_config():
    return Pi0SixHeadConfig(expert_names=SKILLS, action_horizon=50,
                           action_dim=32, max_token_len=200, discrete_state_input=True)


@dataclasses.dataclass(frozen=True)
class RoboCasaV3Data(DataConfigFactory):
    canonical_heads: tuple[str, ...] = SKILLS

    def create(self, assets_dirs, model):
        del assets_dirs
        recipe = source_recipe()
        stats = [normalize.load(Path(recipe[s]['norm_path']).parent) for s in SKILLS]
        for one in stats:
            assert set(one) == {'state', 'actions'}
            for value in one.values():
                assert value.mean.shape == (32,) and value.q01.shape == (32,)
        return DataConfig(repo_id='robocasa_v3_shared_vlm', asset_id='robocasa_v3',
                          norm_stats=stats[0], per_expert_norm_stats=stats,
                          use_per_expert_norm=True, use_quantile_norm=recipe['close']['use_quantile_norm'],
                          skill_segments_canonical_heads=SKILLS,
                          model_transforms=ModelTransformFactory()(model))


def inputs(item, expert):
    state = np.concatenate([item['state.' + key] for key in STATE_KEYS], axis=1)[0]
    actions = np.concatenate([item['action.' + key] for key in ACTION_KEYS], axis=1)
    assert state.shape == (16,) and actions.shape == (50, 12)
    return {'state': transforms.pad_to_dim(state, 32),
            'actions': transforms.pad_to_dim(actions, 32),
            'image': {key: item['video.' + camera][0] for key, camera in CAMERAS.items()},
            'image_mask': {key: np.True_ for key in CAMERAS},
            'prompt': item['annotation.human.task_description'][0],
            'skill_canonical_ids': np.int32(expert)}


class V3Dataset(torch.utils.data.Dataset):
    """Official LeRobot data with deterministic dataset/episode/frame sampling."""

    def __init__(self, seed=42):
        self.seed = seed
        self.recipe = source_recipe(validate_datasets=True)
        self._pools = None
        self._transform = None

    def initialize(self):
        if self._pools is not None:
            return
        self._pools, self.probabilities = [], []
        for name in SKILLS:
            pool = []
            for entry in self.recipe[name]['data_dirs']:
                path = Path(entry['path'])
                for filename in ('info.json', 'modality.json', 'episodes.jsonl', 'tasks.jsonl'):
                    assert (path / 'meta' / filename).is_file(), path / filename
                pool.append(RoboCasaLeRobotDataset(path, filter_key=entry.get('filter_key'), filter_key_seed=0))
            weights = self.recipe[name].get('dataset_weights')
            weights = np.asarray(weights if weights is not None else [len(ds)**0.4 for ds in pool],
                                 dtype=np.float64)
            assert len(weights) == len(pool) and np.all(weights > 0)
            self._pools.append(pool)
            self.probabilities.append(weights / weights.sum())
        config = RoboCasaV3Data().create([], model_config())
        self._transform = transforms.compose([
            PerExpertNormalize(config.per_expert_norm_stats, use_quantiles=config.use_quantile_norm),
            *config.model_transforms.inputs])

    def raw_sample(self, index):
        self.initialize()
        step, expert, slot = index
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, expert, slot]))
        dataset_index = int(rng.choice(len(self._pools[expert]), p=self.probabilities[expert]))
        dataset = self._pools[expert][dataset_index]
        trajectory_index = int(rng.integers(len(dataset.trajectory_ids)))
        trajectory_id = int(dataset.trajectory_ids[trajectory_index])
        frame = int(rng.integers(dataset.trajectory_lengths[trajectory_index]))
        item = dataset.get_step_data(trajectory_id, frame)
        metadata = {'skill': SKILLS[expert], 'dataset': str(dataset.dataset_path),
                    'episode': trajectory_id, 'row': frame,
                    'timestamp': float(dataset.curr_traj_data['timestamp'].iloc[frame])}
        return inputs(item, expert), metadata

    def __getitem__(self, index):
        raw, _ = self.raw_sample(index)
        return self._transform(raw)

    def __len__(self):
        return 45000 * 48


class V3BatchSampler:
    def __init__(self, start=0, stop=45000):
        self.start, self.stop = start, stop

    def __iter__(self):
        for step in range(self.start, self.stop):
            yield [(step, expert, slot) for expert, count in enumerate(counts_at(step))
                   for slot in range(count)]

    def __len__(self):
        return self.stop - self.start


def collate(rows):
    import jax
    return jax.tree.map(lambda *values: np.stack(values), *rows)


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def torch_loader(start=0, stop=45000, workers=2):
    kwargs = dict(dataset=V3Dataset(), batch_sampler=V3BatchSampler(start, stop),
                  num_workers=workers, collate_fn=collate, worker_init_fn=worker_init)
    if workers:
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=True)
    return torch.utils.data.DataLoader(**kwargs)
