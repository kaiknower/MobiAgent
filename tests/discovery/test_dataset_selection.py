from pathlib import Path

from mobiagent.discovery.behavior.dataset_selection import select_first_demo_per_task


def test_select_first_demo_per_task_uses_filename_sort_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "behavior_224_rgb"
    head_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.head"
    left_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.left_wrist"
    right_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.right_wrist"
    meta_dir = dataset_root / "meta" / "episodes" / "task-0001"
    data_dir = dataset_root / "data" / "task-0001"
    for directory in (head_dir, left_dir, right_dir, meta_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (head_dir / "episode_00010005.mp4").write_bytes(b"head")

    for episode_id in ("episode_00010020", "episode_00010010"):
        (head_dir / f"{episode_id}.mp4").write_bytes(b"head")
        (left_dir / f"{episode_id}.mp4").write_bytes(b"left")
        (right_dir / f"{episode_id}.mp4").write_bytes(b"right")
        (meta_dir / f"{episode_id}.json").write_text("{}", encoding="utf-8")
        (data_dir / f"{episode_id}.parquet").write_bytes(b"PAR1")

    selected = select_first_demo_per_task(
        dataset_root=dataset_root,
        task_ids=["task-0001"],
    )

    assert len(selected) == 1
    assert selected[0].episode_id == "episode_00010010"
    assert selected[0].head_video_path.name == "episode_00010010.mp4"
    assert selected[0].left_video_path.name == "episode_00010010.mp4"
    assert selected[0].right_video_path.name == "episode_00010010.mp4"
    assert selected[0].meta_path.name == "episode_00010010.json"
    assert selected[0].parquet_path.name == "episode_00010010.parquet"


def test_select_first_demo_per_task_skips_missing_task_directory(tmp_path: Path) -> None:
    dataset_root = tmp_path / "behavior_224_rgb"
    (dataset_root / "videos").mkdir(parents=True, exist_ok=True)

    selected = select_first_demo_per_task(
        dataset_root=dataset_root,
        task_ids=["task-0001"],
    )

    assert selected == []


def test_select_first_demo_per_task_skips_tasks_without_complete_episodes(tmp_path: Path) -> None:
    dataset_root = tmp_path / "behavior_224_rgb"
    head_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.head"
    left_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.left_wrist"
    right_dir = dataset_root / "videos" / "task-0001" / "observation.images.rgb.right_wrist"
    meta_dir = dataset_root / "meta" / "episodes" / "task-0001"
    data_dir = dataset_root / "data" / "task-0001"
    for directory in (head_dir, left_dir, right_dir, meta_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (head_dir / "episode_00010010.mp4").write_bytes(b"head")
    (left_dir / "episode_00010010.mp4").write_bytes(b"left")

    selected = select_first_demo_per_task(
        dataset_root=dataset_root,
        task_ids=["task-0001"],
    )

    assert selected == []


def test_select_first_demo_per_task_preserves_task_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "behavior_224_rgb"
    for task_id, episode_id in (("task-0002", "episode_00020010"), ("task-0001", "episode_00010010")):
        head_dir = dataset_root / "videos" / task_id / "observation.images.rgb.head"
        left_dir = dataset_root / "videos" / task_id / "observation.images.rgb.left_wrist"
        right_dir = dataset_root / "videos" / task_id / "observation.images.rgb.right_wrist"
        meta_dir = dataset_root / "meta" / "episodes" / task_id
        data_dir = dataset_root / "data" / task_id
        for directory in (head_dir, left_dir, right_dir, meta_dir, data_dir):
            directory.mkdir(parents=True, exist_ok=True)

        (head_dir / f"{episode_id}.mp4").write_bytes(b"head")
        (left_dir / f"{episode_id}.mp4").write_bytes(b"left")
        (right_dir / f"{episode_id}.mp4").write_bytes(b"right")
        (meta_dir / f"{episode_id}.json").write_text("{}", encoding="utf-8")
        (data_dir / f"{episode_id}.parquet").write_bytes(b"PAR1")

    selected = select_first_demo_per_task(
        dataset_root=dataset_root,
        task_ids=["task-0002", "task-0001"],
    )

    assert [demo.task_id for demo in selected] == ["task-0002", "task-0001"]
