"""Constants for the s1_mobile (nokaikai/s1_mobile) offline skill-discovery package.

The on-disk layout for this dataset is:

    <dataset_root>/
        videos/{head_rgb,left_rgb,right_rgb}/<dataset_id>/episode_NNNNNN.mp4
        moving_trash_disposal/<dataset_id>/lerobot_so3_data_30hz/
            data/chunk-000/episode_NNNNNN.parquet
            meta/{info.json,episodes.jsonl,tasks.jsonl}

Each high-level task (general / bottle / can) is backed by multiple dataset_ids
that are physically separate recording batches but share the same natural-language
instruction. The mapping lives in TASK_TO_DATASET_IDS.
"""

# Top-level dataset family directory under which dataset_ids live. A task_id is
# mapped to {family, [dataset_ids]} so different tasks can come from different
# families (e.g. trash_disposal vs pour_blue_particles).
DEFAULT_DATASET_FAMILY_DIRNAME = "moving_trash_disposal"
DATASET_FAMILY_DIRNAME = "moving_trash_disposal"  # kept for backward-compat callers
LEROBOT_SUBDIR = "lerobot_so3_data_30hz"

VIDEOS_DIRNAME = "videos"
META_DIRNAME = "meta"
DATA_DIRNAME = "data"
CHUNK_DIRNAME = "chunk-000"

# Canonical LeRobot v2.0 video_key names (match info.json `features` keys
# whose dtype is "video") — used as the directory name under
# <dataset_id>/<LEROBOT_SUBDIR>/videos/chunk-000/<video_key>/episode_*.mp4
HEAD_CAMERA_DIRNAME = "images_dict.head.rgb"
LEFT_CAMERA_DIRNAME = "images_dict.left.rgb"
RIGHT_CAMERA_DIRNAME = "images_dict.right.rgb"

ACTION_COLUMN = "cartesian_so3_dict.cartesian_pose_command"
STATE_COLUMN = "cartesian_so3_dict.cartesian_pose_state"
JOINT_ACTION_COLUMN = "joints_dict.joints_position_command"
JOINT_STATE_COLUMN = "joints_dict.joints_position_state"
TIMESTAMP_COLUMN = "timestamp"

# Semantic groups inside cartesian_so3 command/state (length 34).
# Aligned with the `names` list in meta/info.json for
# cartesian_so3_dict.cartesian_pose_command.
ACTION_DIM = 34
ACTION_GROUP_SLICES: tuple[tuple[str, slice], ...] = (
    ("torso", slice(0, 9)),
    ("left_arm", slice(9, 18)),
    ("left_gripper", slice(18, 19)),
    ("right_arm", slice(19, 28)),
    ("right_gripper", slice(28, 29)),
    ("head", slice(29, 31)),
    ("chassis", slice(31, 34)),
)
CHASSIS_SLICE = slice(31, 34)

TASK_TO_DATASET_IDS: dict[str, list[str]] = {
    "trash-general": [
        "20260520_Cognition_Moving_trash_disposal_S1_42_1",
        "20260520_Cognition_Moving_trash_disposal_S1_42_2",
    ],
    "trash-bottle": [
        "20260520_Cognition_Moving_trash_disposal_S1_42_bottle_1",
        "20260521_Cognition_Moving_trash_disposal_S1_42_bottle_1",
    ],
    "trash-can": [
        "20260521_Cognition_Moving_trash_disposal_S1_42_can_1",
        "20260521_Cognition_Moving_trash_disposal_S1_42_can_2",
    ],
    "pour-blue": [
        "20260522_Cognition_Moving_pour_blue_particles_S1_47_1",
        "20260522_Cognition_Moving_pour_blue_particles_S1_47_2",
    ],
}

# Per-task family override. Tasks not listed here use DEFAULT_DATASET_FAMILY_DIRNAME.
TASK_TO_FAMILY: dict[str, str] = {
    "pour-blue": "moving_pour_blue_particles",
}

DEFAULT_TASK_IDS = list(TASK_TO_DATASET_IDS.keys())
