"""Read official RoboCasa LeRobot v2 episodes without importing the simulator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STATE_KEYS = ('end_effector_position_relative', 'end_effector_rotation_relative',
              'base_position', 'base_rotation', 'gripper_qpos')
ACTION_KEYS = ('end_effector_position', 'end_effector_rotation', 'gripper_close',
               'base_motion', 'control_mode')
CAMERAS = {'base_0_rgb': 'robot0_agentview_left',
           'left_wrist_0_rgb': 'robot0_eye_in_hand',
           'right_wrist_0_rgb': 'robot0_agentview_right'}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class RoboCasaLeRobotDataset:
    """Metadata-directed state, action, language and RGB loading for one task."""

    def __init__(self, dataset_path, filter_key=None, filter_key_seed=0):
        self.dataset_path = Path(dataset_path)
        self.info = json.loads((self.dataset_path / 'meta/info.json').read_text())
        self.modality = json.loads((self.dataset_path / 'meta/modality.json').read_text())
        episodes = read_jsonl(self.dataset_path / 'meta/episodes.jsonl')
        if filter_key:
            import random
            indices = [x['episode_index'] for x in episodes]
            random.Random(filter_key_seed).shuffle(indices)
            selected = set(indices[:int(filter_key.split('_')[0])])
            episodes = [x for x in episodes if x['episode_index'] in selected]
        if not episodes or any(int(x['length']) <= 0 for x in episodes):
            raise ValueError(f'No nonempty episodes in {self.dataset_path}')
        self.trajectory_ids = np.asarray([int(x['episode_index']) for x in episodes])
        self.trajectory_lengths = np.asarray([int(x['length']) for x in episodes])
        self._lengths = dict(zip(self.trajectory_ids.tolist(), self.trajectory_lengths.tolist(), strict=True))
        self.tasks = {int(x['task_index']): x['task']
                      for x in read_jsonl(self.dataset_path / 'meta/tasks.jsonl')}
        self.curr_traj_id = None
        self.curr_traj_data = None
        self._vectors = {}
        for modality, keys, width in [('state', STATE_KEYS, 16), ('action', ACTION_KEYS, 12)]:
            if any(key not in self.modality.get(modality, {}) for key in keys):
                raise ValueError(f'Missing RoboCasa {modality} modalities in {self.dataset_path}')
            if sum(self.modality[modality][key]['end'] - self.modality[modality][key]['start']
                   for key in keys) != width:
                raise ValueError(f'Expected {width}-dimensional {modality} in {self.dataset_path}')
        for camera in CAMERAS.values():
            if camera not in self.modality.get('video', {}):
                raise ValueError(f'Missing camera {camera} in {self.dataset_path}')

    def __len__(self):
        return int(self.trajectory_lengths.sum())

    def _path(self, pattern, episode, **kwargs):
        chunk_size = self.info.get('chunks_size', 1000)
        relative = pattern.format(episode_chunk=episode // chunk_size, episode_index=episode, **kwargs)
        path = (self.dataset_path / relative).resolve()
        if not path.is_relative_to(self.dataset_path.resolve()):
            raise ValueError(f'Dataset path leaves its root: {relative}')
        return path

    def episode_path(self, episode):
        return self._path(self.info['data_path'], episode)

    def get_trajectory_data(self, episode):
        if self.curr_traj_id != episode:
            table = pd.read_parquet(self.episode_path(episode))
            if len(table) != self._lengths[episode]:
                raise ValueError(f'Episode length mismatch: {self.episode_path(episode)}')
            self.curr_traj_data, self.curr_traj_id = table, episode
            self._vectors = {}
        return self.curr_traj_data

    def vector(self, episode, modality, key):
        table = self.get_trajectory_data(episode)
        cache_key = (modality, key)
        if cache_key not in self._vectors:
            meta = self.modality[modality][key]
            column = meta.get('original_key') or ('observation.state' if modality == 'state' else 'action')
            values = np.stack(table[column].to_numpy()).astype(np.float32)
            values = values[:, meta['start']:meta['end']]
            if values.shape[1] != meta['end'] - meta['start'] or not np.isfinite(values).all():
                raise ValueError(f'Invalid {modality}.{key} values in {self.dataset_path}')
            self._vectors[cache_key] = values
        return self._vectors[cache_key]

    def state_actions(self, episode):
        state = np.concatenate([self.vector(episode, 'state', key) for key in STATE_KEYS], axis=1)
        action = np.concatenate([self.vector(episode, 'action', key) for key in ACTION_KEYS], axis=1)
        return np.pad(state, ((0, 0), (0, 16))), np.pad(action, ((0, 0), (0, 20)))

    def video_path(self, episode, camera):
        key = self.modality['video'][camera].get('original_key') or camera
        return self._path(self.info['video_path'], episode, video_key=key)

    def _frame(self, episode, camera, timestamp):
        import cv2
        path = self.video_path(episode, camera)
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not capture.isOpened() or fps <= 0:
                raise ValueError(f'Cannot read video: {path}')
            index = int(np.rint(float(timestamp) * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Missing video frame {index}: {path}')
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    def get_step_data(self, episode, row):
        table = self.get_trajectory_data(episode)
        if not 0 <= row < len(table):
            raise IndexError(row)
        item = {}
        for modality, keys in [('state', STATE_KEYS), ('action', ACTION_KEYS)]:
            indices = row + np.arange(50 if modality == 'action' else 1)
            for key in keys:
                values = self.vector(episode, modality, key)
                result = values[np.minimum(indices, len(values) - 1)].copy()
                if not self.modality[modality][key].get('absolute', True):
                    result[indices >= len(values)] = 0
                item[f'{modality}.{key}'] = result
        timestamp = float(table['timestamp'].iloc[row])
        for camera in CAMERAS.values():
            item['video.' + camera] = self._frame(episode, camera, timestamp)[None]
        language = self.modality.get('annotation', {}).get('human.task_description', {})
        task_column = language.get('original_key') or 'task_index'
        task_index = int(table[task_column].iloc[row])
        item['annotation.human.task_description'] = [self.tasks[task_index]]
        return item
