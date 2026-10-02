"""OmniGibson environment adapter for BEHAVIOR-1K.

Implements EnvProtocol using OmniGibson and BEHAVIOR gello configuration helpers.
The R1Pro robot provides the 256-dimensional proprioceptive layout used by the
policy. Runtime requires BEHAVIOR-1K, OmniGibson, Isaac Sim and simulator assets.
Initialize Isaac Sim before reset and set OMNI_KIT_ACCEPT_EULA=YES.

Task IDs are translated to activity names through TASK_TO_ACTIVITY.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np


def _to_numpy_or_list(v: Any) -> Any:
    """Convert torch tensor / OG handle / numpy / list to a plain ndarray."""
    if v is None:
        return None
    if hasattr(v, "detach"):  # torch tensor
        return v.detach().cpu().numpy()
    return np.asarray(v)

logger = logging.getLogger("env_omni")


# Map our task-XXXX ID to the official BEHAVIOR activity name (BDDL).
TASK_TO_ACTIVITY: dict[str, str] = {
    "task-0000": "turning_on_radio",
    "task-0001": "picking_up_trash",
    "task-0002": "putting_away_Halloween_decorations",
    "task-0003": "cleaning_up_plates_and_food",
    "task-0004": "can_meat",
    "task-0005": "setting_mousetraps",
    "task-0006": "hiding_Easter_eggs",
    "task-0007": "picking_up_toys",
    "task-0008": "rearranging_kitchen_furniture",
    "task-0009": "putting_up_Christmas_decorations_inside",
    "task-0010": "set_up_a_coffee_station_in_your_kitchen",
    "task-0011": "putting_dishes_away_after_cleaning",
    "task-0012": "preparing_lunch_box",
    "task-0013": "loading_the_car",
    "task-0014": "carrying_in_groceries",
    "task-0015": "bringing_in_wood",
    "task-0016": "moving_boxes_to_storage",
    "task-0017": "bringing_water",
    "task-0018": "tidying_bedroom",
    "task-0019": "outfit_a_basic_toolbox",
    "task-0020": "sorting_vegetables",
    "task-0021": "collecting_childrens_toys",
    "task-0022": "putting_shoes_on_rack",
    "task-0023": "boxing_books_up_for_storage",
    "task-0024": "storing_food",
    "task-0025": "clearing_food_from_table_into_fridge",
    "task-0026": "assembling_gift_baskets",
    "task-0027": "sorting_household_items",
    "task-0028": "getting_organized_for_work",
    "task-0029": "clean_up_your_desk",
    "task-0030": "setting_the_fire",
    "task-0031": "clean_boxing_gloves",
    "task-0032": "wash_a_baseball_cap",
    "task-0033": "wash_dog_toys",
    "task-0034": "hanging_pictures",
    "task-0035": "attach_a_camera_to_a_tripod",
    "task-0036": "clean_a_patio",
    "task-0037": "clean_a_trumpet",
    "task-0038": "spraying_for_bugs",
    "task-0039": "spraying_fruit_trees",
    "task-0040": "make_microwave_popcorn",
    "task-0041": "cook_cabbage",
    "task-0042": "chop_an_onion",
    "task-0043": "slicing_vegetables",
    "task-0044": "cook_hot_dogs",
    "task-0045": "cook_hot_dogs",
    "task-0046": "cook_bacon",
    "task-0047": "freeze_pies",
    "task-0048": "canning_food",
    "task-0049": "make_pizza",
}


def _set_global_macros(headless: bool = True) -> None:
    """Apply the same gm flags the official b1k Evaluator uses."""
    from omnigibson.macros import gm
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = True
    gm.HEADLESS = bool(headless)


class OmniGibsonEnv:
    """EnvProtocol implementation backed by `omnigibson.Environment` + BehaviorTask.

    Lifetime: lazy. Construction is cheap; `reset()` is where OmniGibson
    actually initializes (scene load + asset bake — ~30s on first call).
    """

    def __init__(
        self,
        *,
        instance_id: int = 0,
        headless: bool = True,
        max_episode_steps: int = 5000,
        image_size: int = 224,
    ) -> None:
        self.instance_id = int(instance_id)
        self.headless = bool(headless)
        self.max_episode_steps = int(max_episode_steps)
        # Training data was collected at 224×224. The gello default is 1080×1080
        # (teleop quality); we override to match the model's expected input.
        self.image_size = int(image_size)
        self._env = None
        self._latest_obs: dict[str, Any] | None = None
        self._task_id: str | None = None
        self._episode_steps = 0
        self._episode_terminated = False
        self._episode_truncated = False
        self._episode_step_boundary: int | None = None

    # ---------- EnvProtocol methods ----------

    def reset(self, task: str, *, tro_instance_id: int | None = None) -> dict[str, Any]:
        """Reset to a fresh episode of `task`.

        If `tro_instance_id` is given, a per-instance `tro_state.json` overlay
        is applied AFTER the base reset using the BEHAVIOR evaluator format
        (`omnigibson/learning/eval.py:load_task_instance`).
        Without this overlay, every reset produces the SAME default
        starting state from the base `_0_0_template`.

        See `test_instances.csv` for the canonical 20 public test instance
        ids per task.
        """
        if task not in TASK_TO_ACTIVITY:
            raise ValueError(
                f"Unknown task {task!r}. Add a TASK_TO_ACTIVITY entry or pass "
                f"the activity_name directly via env subclass."
            )
        self._task_id = task
        self._episode_steps = 0
        self._episode_terminated = False
        self._episode_truncated = False
        self._episode_step_boundary = None

        if self._env is None:
            _set_global_macros(headless=self.headless)
            cfg = self._build_config()
            import omnigibson as og
            self._env = og.Environment(configs=cfg)
            # Match training-time head FOV. Training data was rendered with
            # `horizontal_aperture=40.0` on the head zed camera (see
            # OmniGibson/scripts/learning/replay_obs.py:208). The OmniGibson
            # VisionSensor default is 20.995, which gives a ~63.6° horizontal
            # FOV — about 36° narrower than the training 99.6° FOV. Feeding the
            # policy a "zoomed-in" head view at inference is OOD and visibly
            # makes targets look further away than during training. Wrist
            # cameras keep the default aperture (training did NOT override
            # them).
            # Use the BEHAVIOR evaluation pipeline (DepthLowResWrapper +
            # eval_b1k_wrapper.process_obs):
            #   head : aperture=40, render at 720x720
            #   wrist: render at 480x480 (default aperture)
            # Then `resize_with_pad → 224` is applied in `_to_deployment_obs`
            # before the obs reaches the wire / policy. This matches the
            # exact pixel pipeline used at training time.
            try:
                robot = self._env.robots[0]
                head_cam = robot.sensors["robot_r1:zed_link:Camera:0"]
                head_cam.horizontal_aperture = 40.0
                head_cam.image_height = 720
                head_cam.image_width  = 720
                for wrist_name in (
                    "robot_r1:left_realsense_link:Camera:0",
                    "robot_r1:right_realsense_link:Camera:0",
                ):
                    wcam = robot.sensors[wrist_name]
                    wcam.image_height = 480
                    wcam.image_width  = 480
                self._env.load_observation_space()
                logger.info(
                    "BEHAVIOR cameras: head 720x720 aperture=40, "
                    "wrist 480x480 aperture=default; obs resize_with_pad → 224 "
                    "in _to_deployment_obs"
                )
            except Exception:
                logger.exception("failed to override camera resolutions / aperture")
        else:
            # Already loaded — switch task by reloading task definition
            cfg = self._build_config()
            self._env.update_task(cfg["task"])

        # gym-style API: reset() -> (obs, info)
        result = self._env.reset()

        # Apply per-instance tro_state overlay if requested. Mirrors
        # eval.py:load_task_instance — sets robot pose + task-relevant object
        # states from the cached tro_state.json. Returns a fresh obs after
        # physics settling.
        if tro_instance_id is not None:
            self._load_tro_state(int(tro_instance_id))
            result = self._env.reset()

        raw_obs = result[0] if isinstance(result, tuple) else result
        self._latest_obs = self._to_deployment_obs(raw_obs)
        return self._latest_obs

    def _load_tro_state(self, instance_id: int) -> None:
        """Load `<scene>_task_<activity>_0_<id>_template-tro_state.json`
        and apply it to the live env, then settle physics.

        Direct port of `omnigibson/learning/eval.py:load_task_instance`
        in the BEHAVIOR evaluator. Without this step, every reset gives the SAME
        default starting state from the BDDL `_0_0_template`.
        """
        import json as _json
        import os as _os
        import omnigibson as og
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        env = self._env
        if env is None:
            raise RuntimeError("env not initialized")

        scene_model = env.task.scene_name
        tro_filename = env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=env.task.activity_name,
            activity_definition_id=env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = _os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{env.task.activity_name}_instances/"
            f"{tro_filename}-tro_state.json",
        )
        if not _os.path.exists(tro_file_path):
            raise FileNotFoundError(
                f"tro_state file missing for instance_id={instance_id}: "
                f"{tro_file_path}"
            )
        logger.info(
            "loading tro_state instance_id=%d from %s",
            instance_id, tro_file_path,
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(_json.load(f))

        robot = env.scene.robots[0]
        for tro_key, tro_value in tro_state.items():
            if tro_key == "robot_poses":
                # Per-instance presampled robot pose. There can be multiple;
                # eval.py uses [0] (first sampled pose).
                presampled = tro_value
                pose_entry = presampled[robot.model_name][0]
                robot.set_position_orientation(
                    pose_entry["position"], pose_entry["orientation"],
                )
                env.scene.write_task_metadata(key=tro_key, data=tro_value)
            else:
                env.task.object_scope[tro_key].load_state(
                    tro_value, serialized=False,
                )

        # Stabilize: thin / small-mass items can jitter after load_state.
        for _ in range(25):
            og.sim.step_physics()
            for entity in env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        env.scene.update_initial_file()
        env.scene.reset()

    def step(self, action: Any) -> dict[str, Any]:
        """Replay an entire action chunk through the underlying env.

        `action` may be:
          - dict with key "actions": ndarray (T, 23) — server chunk response
          - 2-D ndarray (T, 23) / list-of-lists — bare chunk
          - 1-D ndarray / list — single action

        Stashes every intra-chunk normalized obs in `self._intra_chunk_obs`
        so a video writer / debugger can replay full 30 Hz frames after each
        orchestrator tick (cleared at the start of every step()).
        """
        if self._env is None:
            raise RuntimeError("OmniGibsonEnv.step called before reset()")
        actions = self._extract_actions_chunk(action)
        if not actions:
            raise ValueError("step() received empty action chunk")

        terminated = truncated = False
        info: dict[str, Any] = {}
        last_obs: Any = None
        chunk_steps = 0
        intra: list[dict[str, Any]] = []
        # OmniGibson renders the cameras SLOWER than the physics/action rate —
        # on non-render steps `_to_deployment_obs` produces no camera key, which
        # used to drop ~⅔ of video frames. Carry the last rendered image forward
        # so EVERY env step yields one frame with all 3 cameras present.
        from .policy_client import OPENPI_HEAD_KEY, OPENPI_LEFT_KEY, OPENPI_RIGHT_KEY  # noqa: PLC0415
        _cam_keys = (OPENPI_HEAD_KEY, OPENPI_LEFT_KEY, OPENPI_RIGHT_KEY)
        if not hasattr(self, "_last_cam"):
            self._last_cam: dict[str, Any] = {}
        import omnigibson as og  # noqa: PLC0415
        from time import perf_counter as _pc  # noqa: PLC0415
        from . import timing as _timing  # noqa: PLC0415
        for a in actions:
            if (
                self._episode_step_boundary is not None
                and self._episode_steps + chunk_steps >= self._episode_step_boundary
            ):
                break
            # BEHAVIOR decomposition so RENDER time is measured and
            # EXCLUDED from the reported total: _convert -> _pre_step ->
            # physics (render OFF) -> render(once) -> _post_step. Equivalent to
            # the original `self._env.step(a)` (n_render_iterations=1).
            _act = self._env._convert_action_to_tensor(a)
            self._env._pre_step(_act)
            _t = _pc()
            with og.sim.render_on_step(False):
                og.sim.step()
            _t_phys = _pc() - _t
            _t = _pc()
            og.sim.render()
            _t_render = _pc() - _t
            obs, _reward, terminated, truncated, info = self._env._post_step(_act)
            _timing.add_phys_render(_t_phys, _t_render, 1)
            last_obs = obs
            chunk_steps += 1
            dep = self._to_deployment_obs(obs)
            for k in _cam_keys:
                v = dep.get(k)
                if v is not None:
                    self._last_cam[k] = v          # fresh render — remember it
                elif self._last_cam.get(k) is not None:
                    dep[k] = self._last_cam[k]      # no fresh render — hold last
            intra.append(dep)
            if terminated or truncated:
                break

        self._episode_steps += chunk_steps
        self._episode_terminated = bool(terminated)
        self._episode_truncated = bool(
            truncated or self._episode_steps >= self.max_episode_steps
        )
        self._intra_chunk_obs = intra
        self._latest_obs = intra[-1] if intra else self._to_deployment_obs(last_obs)
        meta = self._latest_obs.setdefault("_meta", {})
        meta["terminated"] = bool(terminated)
        meta["truncated"] = self._episode_truncated
        meta["chunk_steps"] = chunk_steps
        meta["episode_steps"] = self._episode_steps
        meta["episode_step_boundary"] = self._episode_step_boundary
        if isinstance(info, dict):
            meta["info"] = info
        return self._latest_obs

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    def is_done(self) -> bool:
        return (
            self._episode_terminated
            or self._episode_truncated
            or self._episode_steps >= self.max_episode_steps
        )

    def set_episode_step_boundary(self, boundary: int | None) -> None:
        if boundary is not None and boundary <= self._episode_steps:
            raise ValueError(
                f"episode step boundary {boundary} must exceed current step "
                f"{self._episode_steps}"
            )
        self._episode_step_boundary = boundary

    def get_intra_chunk_frames(self, camera: str = "head") -> list[Any]:
        """Frames captured during the most recent `step()`. Camera is one of
        `head` / `left_wrist` / `right_wrist`. Empty list if `step()` not yet called.

        Tries the BEHAVIOR key first (egocentric_camera / wrist_image_*),
        falls back to the legacy head_image / left_wrist_image / right_wrist_image
        names so older recordings still resolve.
        """
        from .policy_client import OPENPI_HEAD_KEY, OPENPI_LEFT_KEY, OPENPI_RIGHT_KEY  # noqa: PLC0415
        intra = getattr(self, "_intra_chunk_obs", None) or []
        candidates = {
            "head":        (OPENPI_HEAD_KEY,  "observation/head_image"),
            "left_wrist":  (OPENPI_LEFT_KEY,  "observation/left_wrist_image"),
            "right_wrist": (OPENPI_RIGHT_KEY, "observation/right_wrist_image"),
        }.get(camera, (OPENPI_HEAD_KEY, "observation/head_image"))
        out = []
        for o in intra:
            for k in candidates:
                if k in o:
                    out.append(o[k]); break
        return out

    def get_camera_frame(self, camera: str = "head") -> Any:
        if self._latest_obs is None:
            return None
        from .policy_client import OPENPI_HEAD_KEY, OPENPI_LEFT_KEY, OPENPI_RIGHT_KEY  # noqa: PLC0415
        candidates = {
            "head":        (OPENPI_HEAD_KEY,  "observation/head_image"),
            "left_wrist":  (OPENPI_LEFT_KEY,  "observation/left_wrist_image"),
            "right_wrist": (OPENPI_RIGHT_KEY, "observation/right_wrist_image"),
        }.get(camera, (OPENPI_HEAD_KEY, "observation/head_image"))
        for k in candidates:
            v = self._latest_obs.get(k)
            if v is not None:
                return v
        return None

    def is_success(self) -> bool:
        if self._env is None:
            return False
        try:
            task = getattr(self._env, "task", None)
            if task is None:
                return False
            return bool(getattr(task, "success", False))
        except Exception:
            logger.exception("is_success check failed")
            return False



    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.exception("close failed")
            self._env = None

    # ---------- internals ----------

    def _build_config(self) -> dict[str, Any]:
        """Mirror behavior-1k-solution/dimos_pi0_5GT/connection.py:243-251.

        Uses gello's task/robot config helpers. The proprio_obs list MUST
        match PROPRIOCEPTION_INDICES["R1Pro"] in eval_utils.py byte-for-byte
        (= the layout used to collect training data) so server-side state
        slicing aligns.
        """
        from gello.robots.sim_robot.og_teleop_utils import (  # type: ignore
            load_available_tasks, generate_robot_config,
        )
        from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES  # type: ignore
        from omnigibson.learning.utils.eval_utils import (
            PROPRIOCEPTION_INDICES, generate_basic_environment_config,
        )

        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available = load_available_tasks()
        activity = TASK_TO_ACTIVITY[self._task_id]
        if activity not in available:
            raise ValueError(
                f"BEHAVIOR activity {activity!r} (mapped from {self._task_id!r}) "
                f"not in load_available_tasks(). Got: {sorted(available.keys())[:10]}..."
            )
        task_cfg = available[activity][0]

        env_cfg = generate_basic_environment_config(task_name=activity, task_cfg=task_cfg)
        env_cfg["robots"] = [generate_robot_config(task_name=activity, task_cfg=task_cfg)]
        env_cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        env_cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())
        env_cfg["robots"][0]["grasping_mode"] = "assisted"
        # Force 224×224 RGB (training-data resolution). gello defaults to 1080.
        env_cfg["robots"][0]["sensor_config"] = {
            "VisionSensor": {
                "sensor_kwargs": {
                    "image_height": self.image_size,
                    "image_width":  self.image_size,
                },
            },
        }
        env_cfg["task"]["termination_config"]["max_steps"] = self.max_episode_steps
        env_cfg["task"]["activity_instance_id"] = self.instance_id
        env_cfg["task"]["include_obs"] = False
        return env_cfg

    @staticmethod
    def _extract_actions_chunk(action: Any) -> list[Any]:
        if isinstance(action, dict):
            arr = action.get("actions")
            if arr is None:
                arr = action.get("action")
            if arr is None:
                raise ValueError(
                    f"action dict has no 'actions'/'action' key (keys={list(action.keys())})"
                )
        else:
            arr = action
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            return [a]
        if a.ndim == 2:
            return [a[i] for i in range(a.shape[0])]
        raise ValueError(f"unexpected action shape: {a.shape}")

    def _to_deployment_obs(self, raw_obs: Any) -> dict[str, Any]:
        """Apply b1k flatten + remap to deployment schema.

        Keeps both the raw flattened dict (for VLM judge / debugging) and
        the deployment keys (`observation/head_image`, etc.).
        """
        from omnigibson.learning.utils.eval_utils import flatten_obs_dict

        out: dict[str, Any] = {"task": self._task_id}
        if not isinstance(raw_obs, dict):
            return out

        flat = flatten_obs_dict(raw_obs)
        # build_policy_observation does the same key→key remap our wire client
        # uses, but the returned dict is what the SERVER sees. We want the
        # ORCHESTRATOR-side latest_obs to carry the same keys (so judge/run
        # can pull head_image), so import its helpers and reuse them.
        from .policy_client import (
            HEAD_RGB_KEY, LEFT_RGB_KEY, RIGHT_RGB_KEY, PROPRIO_KEY,
            OPENPI_HEAD_KEY, OPENPI_LEFT_KEY, OPENPI_RIGHT_KEY, OPENPI_STATE_KEY,
            _find_first, _normalize_image, _to_numpy,
        )

        head  = _find_first(flat, (HEAD_RGB_KEY,))
        left  = _find_first(flat, (LEFT_RGB_KEY,))
        right = _find_first(flat, (RIGHT_RGB_KEY,))
        state = _find_first(flat, (PROPRIO_KEY,))

        # BEHAVIOR pipeline (b1k.shared.eval_b1k_wrapper.process_obs):
        # source 720x720 (head) / 480x480 (wrist) → drop alpha [..., :3] →
        # openpi_client.image_tools.resize_with_pad(224, 224). Same function
        # the server's model_transforms (ResizeImages) runs, so client/server
        # see bit-identical 224x224 letterboxed tensors.
        from openpi_client.image_tools import resize_with_pad as _resize_with_pad  # noqa: PLC0415

        def _to_uint8_hwc(value):
            arr = np.asarray(_to_numpy(value))
            if arr.ndim >= 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
                arr = np.transpose(arr, (1, 2, 0))   # CHW → HWC (rare; OmniGibson is HWC)
            if arr.ndim >= 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]                    # drop alpha (retain RGB channels)
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            return arr

        if head is not None:
            head_full = _to_uint8_hwc(head)                              # 720x720x3 uint8
            out[OPENPI_HEAD_KEY] = _resize_with_pad(head_full, 224, 224)
            out["observation/head_image_orig"] = head_full
        if left is not None:
            left_full = _to_uint8_hwc(left)
            out[OPENPI_LEFT_KEY] = _resize_with_pad(left_full, 224, 224)
            out["observation/left_wrist_image_orig"] = left_full
        if right is not None:
            right_full = _to_uint8_hwc(right)
            out[OPENPI_RIGHT_KEY] = _resize_with_pad(right_full, 224, 224)
            out["observation/right_wrist_image_orig"] = right_full
        if state is not None:
            arr = _to_numpy(state).astype(np.float32, copy=False)
            if arr.ndim > 1:
                arr = arr.reshape(-1)
            out[OPENPI_STATE_KEY] = arr

        # Keep the flattened raw form available for downstream debugging /
        # the VLM judge's image lookup if extra keys are needed.
        out["_flat_obs_keys"] = sorted(flat.keys())
        return out


__all__ = ["OmniGibsonEnv", "TASK_TO_ACTIVITY"]
