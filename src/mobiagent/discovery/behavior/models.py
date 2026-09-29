from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SelectedDemo:
    task_id: str
    episode_id: str
    head_video_path: Path
    left_video_path: Path
    right_video_path: Path
    meta_path: Path
    parquet_path: Path
