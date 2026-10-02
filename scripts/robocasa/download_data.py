"""Download official RoboCasa datasets and record their six-expert task mapping."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ('close', 'open', 'switch', 'manipulate', 'navigate', 'pnp')


def selected_tasks(selection):
    groups = selection['skills']
    if set(groups) != set(SKILLS) or any(not groups[name] for name in SKILLS):
        raise ValueError(f'Define a nonempty task list for each expert: {SKILLS}')
    pairs = [(skill, task) for skill in SKILLS for task in groups[skill]]
    names = [task for _, task in pairs]
    if len(names) != len(set(names)):
        raise ValueError('A task may belong to only one expert')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=ROOT / 'configs/robocasa/datasets.example.json')
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'datasets/robocasa')
    parser.add_argument('--dry-run', action='store_true', help='List official destinations without downloading')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    pairs = selected_tasks(selection)
    root = args.dataset_root.expanduser().resolve()
    try:
        import robocasa.macros as macros
        macros.DATASET_BASE_PATH = str(root)
        from robocasa.utils.dataset_registry_utils import get_ds_meta
        downloader = importlib.import_module('robocasa.scripts.download_datasets')
    except ImportError as exc:
        raise RuntimeError('Run in a RoboCasa 1.0 environment; see docs/robocasa.md#training-data') from exc
    # Both the registry and official downloader must use the same destination.
    downloader.DATASET_BASE_PATH = str(root)
    source, split = selection['source'], selection['split']
    if source not in ('human', 'mimicgen') or split not in ('pretrain', 'target'):
        raise ValueError('source must be human/mimicgen; split must be pretrain/target')
    entries = []
    for skill, task in pairs:
        meta = get_ds_meta(task=task, split=split, source='mg' if source == 'mimicgen' else source)
        if meta is None:
            raise ValueError(f'No official dataset registered for {task}/{split}/{source}')
        path = Path(meta['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'Official dataset destination is outside {root}')
        entries.append({'skill': skill, 'task': task, 'path': str(path), 'filter_key': None})
        print(f'{skill:12s} {task:36s} {path}', flush=True)
    if args.dry_run:
        return
    manifest = root / 'datasets.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['datasets'] != entries:
            raise FileExistsError(f'{manifest} contains another selection; use a separate dataset root')
    downloader.download_datasets(split=[split], tasks=[task for _, task in pairs], source=[source])
    for entry in entries:
        path = Path(entry['path'])
        for name in ('info.json', 'modality.json', 'episodes.jsonl', 'tasks.jsonl'):
            if not (path / 'meta' / name).is_file():
                raise FileNotFoundError(f'Download incomplete: {path / "meta" / name}; rerun after repairing this task')
        if not any(path.glob('data/*/*.parquet')) or not any(path.glob('videos/**/*.mp4')):
            raise FileNotFoundError(f'Missing episode data or videos: {path}')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'source': source, 'split': split,
        'upstream': 'https://github.com/robocasa/robocasa', 'datasets': entries}, indent=2) + '\n')
    print(f'Dataset manifest: {manifest}', flush=True)


if __name__ == '__main__':
    main()
