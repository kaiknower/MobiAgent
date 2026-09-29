"""Policy server client — talks to an openpi serve_policy server over
msgpack/websocket.

Server-side payload contract (3-head S1 model):
    observation/egocentric_camera   (H, W, 3) uint8
    observation/wrist_image_left    (H, W, 3) uint8
    observation/wrist_image_right   (H, W, 3) uint8
    observation/state               (state_dim,) float32
    prompt                          "<bare skill string>"
    stage_override                  int ∈ {0, 1, 2}    (per stage_hint)

Server response: {"actions": (T, action_dim) float32, ...timing/state...}.
Accepts both "action" (single-frame) and "actions" (chunk) keys for
backward compatibility.
"""
from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Mapping, Protocol

import msgpack
import numpy as np

from .schemas import STAGE_HINT_TO_INT

logger = logging.getLogger("mobiagent.robots.s1.policy_client")


# ---------- inlined msgpack-numpy ----------

def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array, use_bin_type=True)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)


# ---------- obs payload ----------

# Canonical openpi-side keys
OPENPI_HEAD_KEY  = "observation/egocentric_camera"
OPENPI_LEFT_KEY  = "observation/wrist_image_left"
OPENPI_RIGHT_KEY = "observation/wrist_image_right"
OPENPI_STATE_KEY = "observation/state"


# Cartesian dims for the s1_mobile EE-pose interface.
#   34-dim raw layout (state and robot-side action):
#     0-8 torso, 9-17 left_arm, 18 left_grip, 19-27 right_arm, 28 right_grip,
#     29 head_yaw, 30 head_pitch, 31-33 chassis
# The 3-head s1_mobile model is trained without the two head dims, so it
# emits a 32-dim action vector (head dims dropped). `_expand_action_dim`
# below re-inserts head_yaw / head_pitch at idx 29-30 so the chunk handed
# to the robot driver is always 34-dim.
_S1_MOBILE_MODEL_ACTION_DIM = 32
_S1_MOBILE_ROBOT_ACTION_DIM = 34
_S1_HEAD_YAW_IDX = 29
_S1_HEAD_PITCH_IDX = 30
# Chassis (base) indices in the raw 34-dim cartesian state. The 3-head
# s1_mobile training zeroes these dims before they reach the model — the
# model never sees absolute base position. Mirror that at inference time
# so the model gets the same distribution it was trained on.
_S1_CHASSIS_SLICE = slice(31, 34)


def _zero_chassis_in_state(state: np.ndarray) -> np.ndarray:
    """Zero the chassis (base) dims in the 34-dim cartesian state vector.

    Returns a copy with state[31:34] set to 0. No-op for shorter vectors.
    """
    if state.ndim != 1 or state.shape[-1] < 34:
        return state
    out = state.astype(np.float32, copy=True)
    out[_S1_CHASSIS_SLICE] = 0.0
    return out


def _expand_action_dim(actions: np.ndarray, *, current_state: np.ndarray | None) -> np.ndarray:
    """Re-insert head_yaw / head_pitch into a 32-dim s1_mobile model action.

    The 3-head s1_mobile training drops head dims (idx 29, 30) before
    feeding the model and outputs a 32-dim action chunk. The robot driver
    still expects the original 34-dim cartesian layout, so we splice the
    head channels back in:

      * if `current_state` is provided and has shape[-1] >= 31, copy
        `state[29:31]` into every action timestep — keeps head at its
        most recent pose
      * else fill 0 — head stays at the zero pose

    No-op when ``actions`` is not 2-D with last dim == 32.
    """
    if actions.ndim != 2 or actions.shape[-1] != _S1_MOBILE_MODEL_ACTION_DIM:
        return actions
    T = actions.shape[0]
    head = np.zeros((T, 2), dtype=actions.dtype)
    if current_state is not None and current_state.ndim == 1 and current_state.shape[-1] >= 31:
        head[:, 0] = float(current_state[_S1_HEAD_YAW_IDX])
        head[:, 1] = float(current_state[_S1_HEAD_PITCH_IDX])
    return np.concatenate([actions[:, :29], head, actions[:, 29:]], axis=1)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_image(value: Any) -> np.ndarray:
    """uint8 HWC numpy. Rotates CHW → HWC, drops alpha, casts float → uint8."""
    image = np.asarray(_to_numpy(value))
    if image.ndim >= 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim >= 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim >= 3 and image.shape[0] == 4:
        image = np.transpose(image[:3], (1, 2, 0))
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _find_first(obs: Mapping[str, Any], candidates: tuple[str, ...]) -> Any:
    """Return the first present non-None value for any of the candidate keys."""
    for k in candidates:
        if k in obs and obs[k] is not None:
            return obs[k]
    return None


def build_policy_observation(
    obs: Mapping[str, Any], *, prompt: str, stage_hint: str | int | None,
) -> dict[str, Any]:
    """Build the dict that goes over the wire.

    Accepts both the openpi canonical keys
    (`observation/egocentric_camera`, `observation/wrist_image_left/right`)
    and the legacy short names (`head_image`, `left_wrist_image`,
    `right_wrist_image`) so older clients still work.
    """
    head  = _find_first(obs, (OPENPI_HEAD_KEY,  "observation/head_image",         "head_image"))
    left  = _find_first(obs, (OPENPI_LEFT_KEY,  "observation/left_wrist_image",   "left_wrist_image"))
    right = _find_first(obs, (OPENPI_RIGHT_KEY, "observation/right_wrist_image",  "right_wrist_image"))
    state = _find_first(obs, (OPENPI_STATE_KEY, "observation/proprio",            "proprio"))

    if head is None or left is None or right is None or state is None:
        present = sorted(obs.keys()) if isinstance(obs, Mapping) else []
        raise ValueError(
            "Missing required obs fields. "
            f"head={head is not None} left={left is not None} "
            f"right={right is not None} state={state is not None}. "
            f"obs keys: {present[:20]}{'...' if len(present) > 20 else ''}"
        )

    head_img  = _normalize_image(head)
    left_img  = _normalize_image(left)
    right_img = _normalize_image(right)
    state_arr = _to_numpy(state).astype(np.float32, copy=False)
    if state_arr.ndim > 1:
        state_arr = state_arr.reshape(-1)
    # Match the training-time state distribution: the 3-head s1_mobile
    # model is trained with chassis (base) dims zeroed before it sees the
    # state, so do the same here. (The server-side input chain also runs
    # _ZeroChassisInState, but doing it client-side keeps the wire payload
    # consistent and removes any ambiguity about what the model actually
    # consumed.)
    state_arr = _zero_chassis_in_state(state_arr)

    payload: dict[str, Any] = {
        OPENPI_HEAD_KEY:  head_img,
        OPENPI_LEFT_KEY:  left_img,
        OPENPI_RIGHT_KEY: right_img,
        OPENPI_STATE_KEY: state_arr,
        "prompt": str(prompt),
    }


    if stage_hint is not None:
        if isinstance(stage_hint, str):
            if stage_hint not in STAGE_HINT_TO_INT:
                raise ValueError(f"Unknown stage_hint: {stage_hint}")
            stage_int = STAGE_HINT_TO_INT[stage_hint]
        else:
            stage_int = int(stage_hint)
        payload["stage_override"] = stage_int
        payload["stage_hint"] = stage_int

    return payload


# ---------- websocket transport ----------


class _Transport:
    """Minimal msgpack-over-websocket client for openpi serve_policy.

    Lazy connect, reads server metadata on first connect, one-shot
    reconnect on a dropped connection.
    """

    def __init__(
        self, *, host: str, port: int, path: str = "/ws",
        api_key: str | None = None,
        connect_timeout_s: float = 30.0,
        infer_timeout_s: float = 60.0,
        secure: bool = False,
    ) -> None:
        scheme = "wss" if secure else "ws"
        self._uri = (
            f"{scheme}://{host}:{port}{path}"
            if path and path != "/" else f"{scheme}://{host}:{port}"
        )
        self._secure = secure
        self._api_key = api_key
        self._connect_timeout_s = connect_timeout_s
        self._infer_timeout_s = infer_timeout_s
        self._packer = _Packer()
        self._ws = None
        self._metadata: dict | None = None

    @property
    def server_metadata(self) -> dict | None:
        return self._metadata

    def _connect(self) -> None:
        try:
            import inspect
            import websockets.sync.client as ws_client
        except Exception as exc:
            raise RuntimeError(
                "policy_client requires `websockets>=11`; pip install websockets"
            ) from exc
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        kwargs: dict = dict(
            compression=None, max_size=None,
            additional_headers=headers,
            open_timeout=self._connect_timeout_s,
            ping_interval=60, ping_timeout=300,
        )
        # `proxy=None` MUST be explicit on websockets ≥15 — without it the
        # lib auto-detects HTTP_PROXY/etc. and silently rewrites the upgrade,
        # which the openpi server closes without responding. Exception: when
        # `secure=True` (wss:// to a public host) the request may legitimately
        # need to traverse HTTP_PROXY/HTTPS_PROXY, so leave `proxy` unset.
        if not self._secure:
            kwargs["proxy"] = None
        _accepted = inspect.signature(ws_client.connect).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _accepted.values()):
            _dropped = [k for k in kwargs if k not in _accepted]
            for k in _dropped:
                kwargs.pop(k)
            if _dropped:
                logger.warning("websockets.connect: dropped unsupported kwargs %s", _dropped)
        ws = ws_client.connect(self._uri, **kwargs)
        try:
            meta = ws.recv()
            self._metadata = _unpackb(meta if isinstance(meta, bytes) else meta.encode("latin-1"))
        except Exception as exc:
            logger.warning("server metadata read failed: %s", exc)
            self._metadata = None
        self._ws = ws

    def infer(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        if self._ws is None:
            self._connect()
        data = self._packer.pack(dict(obs))
        try:
            self._ws.send(data)  # type: ignore[union-attr]
            raw = self._ws.recv(timeout=self._infer_timeout_s)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("websocket failure (%s); reconnecting", exc)
            try:
                self._ws.close()  # type: ignore[union-attr]
            except Exception:
                pass
            self._ws = None
            self._connect()
            self._ws.send(data)  # type: ignore[union-attr]
            raw = self._ws.recv(timeout=self._infer_timeout_s)  # type: ignore[union-attr]
        if isinstance(raw, str):
            raise RuntimeError(f"server returned error frame:\n{raw}")
        return _unpackb(raw)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None


# ---------- public clients ----------


class PolicyClientProtocol(Protocol):
    def request_chunk(
        self, *, obs: dict[str, Any], prompt: str,
        stage_hint: str | int | None = None, timeout_s: float = 30.0,
    ) -> dict[str, Any]: ...

    def close(self) -> None: ...


class WebsocketPolicyClient:
    """openpi-compatible policy client for the S1-mobile 3-head server.

    Layout choices, per `configs/policy_servers.yaml.example`:
      - `shared:` mode — one server hosts all 3 heads, route via `stage_hint`.
      - legacy per-head mode — each `stage_hint` pinned to its own port via
        `bound_stage_hint`.

    The optional rolling-inpainting prior keeps the OT-flow-matching
    denoiser anchored across consecutive chunks. Disabled by default;
    enable via `OPENPI_DISABLE_INPAINT=0` and pass `enable_inpaint=True`.
    """

    def __init__(
        self, *, host: str, port: int, path: str = "/ws",
        bound_stage_hint: str | None = None,
        api_key: str | None = None,
        connect_timeout_s: float = 30.0,
        infer_timeout_s: float = 60.0,
        secure: bool = False,
        enable_inpaint: bool | None = None,
        actions_to_execute: int = 26,
        inpaint_overlap: int = 4,
        inpaint_until_time: float = 0.3,
    ) -> None:
        if enable_inpaint is None:
            # Default: OFF. Set env `OPENPI_DISABLE_INPAINT=0` to opt in.
            enable_inpaint = os.environ.get("OPENPI_DISABLE_INPAINT", "1") != "1"
        self.host = host
        self.port = port
        self.path = path
        self.bound_stage_hint = bound_stage_hint
        self._transport = _Transport(
            host=host, port=port, path=path, api_key=api_key,
            connect_timeout_s=connect_timeout_s,
            infer_timeout_s=infer_timeout_s,
            secure=secure,
        )
        self._enable_inpaint = bool(enable_inpaint)
        self._actions_to_execute = int(actions_to_execute)
        self._inpaint_overlap = int(inpaint_overlap)
        self._inpaint_until_time = float(inpaint_until_time)
        self._prev_actions_raw: np.ndarray | None = None

    @property
    def server_metadata(self) -> dict | None:
        return self._transport.server_metadata

    def request_chunk(
        self, *, obs: dict[str, Any], prompt: str,
        stage_hint: str | int | None = None, timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        if (self.bound_stage_hint and stage_hint is not None
                and stage_hint != self.bound_stage_hint):
            raise ValueError(
                f"WebsocketPolicyClient bound to stage_hint={self.bound_stage_hint!r} "
                f"but received request for {stage_hint!r}. Registry routing bug."
            )
        effective = stage_hint if stage_hint is not None else self.bound_stage_hint
        payload = build_policy_observation(obs, prompt=prompt, stage_hint=effective)

        if self._enable_inpaint and self._prev_actions_raw is not None:
            chunk_len, action_dim = self._prev_actions_raw.shape
            overlap = min(
                self._inpaint_overlap,
                max(0, chunk_len - self._actions_to_execute),
            )
            if overlap > 0:
                prior = np.zeros((chunk_len, action_dim), dtype=np.float32)
                mask = np.zeros((chunk_len,), dtype=bool)
                tail = self._prev_actions_raw[
                    self._actions_to_execute:self._actions_to_execute + overlap
                ]
                prior[:overlap] = tail
                mask[:overlap] = True
                payload["prev_actions"] = prior
                payload["prev_actions_mask"] = mask
                payload["inpaint_until_time"] = self._inpaint_until_time

        if timeout_s and timeout_s != self._transport._infer_timeout_s:
            self._transport._infer_timeout_s = float(timeout_s)
        result = self._transport.infer(payload)

        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)
        elif "action" in result:
            actions = np.asarray(result["action"], dtype=np.float32)
        else:
            raise RuntimeError(
                f"server response missing 'action(s)' key; got keys={list(result.keys())}"
            )
        if actions.ndim == 1:
            actions = actions[None, :]

        # 3-head s1_mobile model emits 32-dim actions (no head dims). Splice
        # head_yaw / head_pitch back at idx 29/30 (sourced from the most
        # recent observation state so the head holds its current pose).
        if actions.shape[-1] == _S1_MOBILE_MODEL_ACTION_DIM:
            actions = _expand_action_dim(actions, current_state=payload.get(OPENPI_STATE_KEY))

        if self._enable_inpaint and actions.ndim == 2:
            self._prev_actions_raw = actions.copy()

        if self._enable_inpaint and len(actions) > self._actions_to_execute:
            actions = actions[:self._actions_to_execute]

        result["actions"] = actions
        return result

    def reset_inpaint_state(self) -> None:
        """Clear the saved prev_actions prior. Call at episode start or when
        the active subtask/prompt changes."""
        self._prev_actions_raw = None

    def close(self) -> None:
        self._transport.close()


class MockPolicyClient:
    """In-process stub for unit tests; matches the real server response shape."""

    def __init__(self, *, stage_hint: str, chunk_size: int = 30, action_dim: int = 34) -> None:
        self.stage_hint = stage_hint
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.history: list[dict[str, Any]] = []

    def request_chunk(
        self, *, obs: dict[str, Any], prompt: str,
        stage_hint: str | int | None = None, timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        time.sleep(0.005)
        hint_used = stage_hint if stage_hint is not None else self.stage_hint
        self.history.append({
            "stage_hint": hint_used,
            "prompt": prompt,
            "obs_step": obs.get("step") if isinstance(obs, dict) else None,
        })
        return {
            "actions": np.zeros((self.chunk_size, self.action_dim), dtype=np.float32),
            "stage_hint_used": hint_used,
            "prompt_echo": prompt,
        }

    def close(self) -> None:
        pass


__all__ = [
    "PolicyClientProtocol",
    "WebsocketPolicyClient",
    "MockPolicyClient",
    "build_policy_observation",
    "OPENPI_HEAD_KEY",
    "OPENPI_LEFT_KEY",
    "OPENPI_RIGHT_KEY",
    "OPENPI_STATE_KEY",
]
