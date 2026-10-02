"""Compute RoboCasa normalization assets and generate a joint-training recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ('close', 'open', 'switch', 'manipulate', 'navigate', 'pnp')


def prepare(manifest_path, output, assets_dir, base_params):
    from openpi.shared import normalize
    from openpi.training.robocasa_lerobot import RoboCasaLeRobotDataset
    manifest = json.loads(manifest_path.read_text())
    grouped = {name: [] for name in SKILLS}
    for entry in manifest['datasets']:
        if entry['skill'] not in grouped:
            raise ValueError(f'Unknown expert: {entry["skill"]}')
        grouped[entry['skill']].append(entry)
    if any(not entries for entries in grouped.values()):
        raise ValueError('Dataset manifest must cover all six experts')
    if output.exists():
        raise FileExistsError(f'Recipe already exists: {output}; use a new --output and --assets-dir')
    if any((assets_dir / name / 'norm_stats.json').exists() for name in SKILLS):
        raise FileExistsError(f'Normalization assets already exist: {assets_dir}; use a new --assets-dir')
    if '://' not in base_params:
        base_params = str(Path(base_params).expanduser().resolve())
        if not Path(base_params).is_dir():
            raise FileNotFoundError(f'Base parameter directory not found: {base_params}')
    recipe = {'skills': {}}
    statistics = {}
    for name in SKILLS:
        state_stats, action_stats = normalize.RunningStats(), normalize.RunningStats()
        data_dirs = []
        for entry in grouped[name]:
            path = Path(entry['path']).expanduser().resolve()
            dataset = RoboCasaLeRobotDataset(path, filter_key=entry.get('filter_key'))
            for episode in dataset.trajectory_ids:
                state, action = dataset.state_actions(int(episode))
                state_stats.update(state.astype(np.float64))
                action_stats.update(action.astype(np.float64))
                # Validate camera coverage without decoding the entire video corpus.
                for camera in dataset.modality['video']:
                    if not dataset.video_path(int(episode), camera).is_file():
                        raise FileNotFoundError(dataset.video_path(int(episode), camera))
            data_dirs.append({'path': str(path), 'filter_key': entry.get('filter_key')})
        directory = assets_dir / name
        statistics[name] = {'state': state_stats.get_statistics(), 'actions': action_stats.get_statistics()}
        recipe['skills'][name] = {'base_params': base_params, 'norm_path': str(directory / 'norm_stats.json'),
            'batch_size': 8, 'action_horizon': 50, 'use_quantile_norm': False,
            'dataset_weights': None, 'data_dirs': data_dirs}
        print(f'{name}: computed normalization from {len(data_dirs)} datasets', flush=True)
    for name in SKILLS:
        normalize.save(assets_dir / name, statistics[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(recipe, indent=2) + '\n')
    print(f'Training recipe: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'datasets/robocasa/datasets.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'configs/robocasa/data.json')
    parser.add_argument('--assets-dir', type=Path, default=ROOT / 'assets/robocasa')
    parser.add_argument('--base-params', default='gs://openpi-assets/checkpoints/pi05_base/params',
                        help='OpenPI pi0.5 base parameters: local directory or supported storage URI')
    args = parser.parse_args()
    prepare(args.manifest.expanduser().resolve(), args.output.expanduser().resolve(),
            args.assets_dir.expanduser().resolve(), args.base_params)


if __name__ == '__main__':
    main()
