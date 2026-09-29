"""Policy server client — talks to openpi serve_policy over msgpack/websocket.

Wire transport + obs payload construction follow the proven recipe from:
  - behavior-1k-solution/dimos_pi0_5GT/pi05_policy_client.py    (Pi05WebsocketTransport)
  - behavior-1k-solution/dimos_pi0_5GT/pi05_policy_bridge.py    (build_policy_observation)

We do NOT import from dimos_pi0_5GT (plan rule: DIMOS-free) but copy the
non-DIMOS parts inline. Those files only depend on numpy + websockets +
msgpack — no dimos.* imports — so the rule is preserved in spirit.

Server-side payload contract (consumed by SkillSegmentInputs / BehaviorInputs
on the openpi side):
    observation/head_image       (H, W, 3) uint8
    observation/left_wrist_image (H, W, 3) uint8
    observation/right_wrist_image(H, W, 3) uint8
    observation/state            (256,)    float32
    prompt                       "<task>. Now: <skill>."
    stage_override               int 0..5    (== stage_hint, server side calls it
                                              stage_override; six-head model reads
                                              it via StageHintToSkillCanonicalId)

Server response: {"actions": (T, 23) float32, ...other timing/state keys...}.
We accept either "action" (legacy single-frame) or "actions" (chunk).
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

logger = logging.getLogger("policy_client")


# ---------- inlined msgpack-numpy (from openpi-client/openpi_client/msgpack_numpy.py) ----------

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


# ---------- obs payload (adapted from pi05_policy_bridge.build_policy_observation) ----------

# Flat-obs keys produced by `flatten_obs_dict` on a real OmniGibson reset:
HEAD_RGB_KEY  = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
LEFT_RGB_KEY  = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
RIGHT_RGB_KEY = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
PROPRIO_KEY   = "robot_r1::proprio"

# Champion-aligned obs keys (b1k.shared.eval_b1k_wrapper.process_obs). Server's
# BehaviorInputs._get_any() accepts both these and the older head_image/
# left_wrist_image/right_wrist_image names, so this is a forward-compatible
# rename — old recordings / probes still resolve.
OPENPI_HEAD_KEY  = "observation/egocentric_camera"
OPENPI_LEFT_KEY  = "observation/wrist_image_left"
OPENPI_RIGHT_KEY = "observation/wrist_image_right"
OPENPI_STATE_KEY = "observation/state"


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_image(value: Any) -> np.ndarray:
    """uint8 HWC numpy. Rotates CHW→HWC, drops alpha, casts float→uint8."""
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
    """Return the first present non-None value for any of the candidate keys.

    Falls back to substring scan against obs keys if no exact match — handles
    OmniGibson's per-instance robot-name suffix (e.g. `robot_nojlwa::...::rgb`)
    where the literal `robot_r1::...` key may not exist.
    """
    for k in candidates:
        if k in obs and obs[k] is not None:
            return obs[k]
    # substring fallback
    for k in candidates:
        # extract a distinctive token from each candidate
        token = k.split("::")[-2] if "::" in k else k  # e.g. "zed_link:Camera:0"
        for obs_k, v in obs.items():
            if token and token in obs_k and v is not None:
                if "rgb" in k and not str(obs_k).endswith("rgb"):
                    continue
                return v
    return None


def build_policy_observation(
    obs: Mapping[str, Any], *, prompt: str, stage_hint: str | int | None,
) -> dict[str, Any]:
    """Build the dict that goes over the wire. Mirrors pi05_policy_bridge but
    also stamps prompt + stage_override (= stage_hint as int)."""
    # Accept champion names (egocentric_camera / wrist_image_left/right) plus
    # the legacy head_image / left_wrist_image / right_wrist_image names so
    # in-flight artifacts and older clients still work.
    head = _find_first(obs, (OPENPI_HEAD_KEY,  "observation/head_image",         HEAD_RGB_KEY))
    left = _find_first(obs, (OPENPI_LEFT_KEY,  "observation/left_wrist_image",   LEFT_RGB_KEY))
    right = _find_first(obs, (OPENPI_RIGHT_KEY, "observation/right_wrist_image", RIGHT_RGB_KEY))
    state = _find_first(obs, (OPENPI_STATE_KEY, PROPRIO_KEY))

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


# ---------- websocket transport (inlined from pi05_policy_client.Pi05WebsocketTransport) ----------


class _Pi05Transport:
    """Minimal msgpack-over-websocket client for openpi serve_policy.

    Lazy connect, server-metadata read on first connect, one-shot reconnect on
    a dropped connection.
    """

    def __init__(
        self, *, host: str, port: int, path: str = "/ws",
        api_key: str | None = None,
        connect_timeout_s: float = 30.0,
        infer_timeout_s: float = 60.0,
        secure: bool = False,
    ) -> None:
        scheme = "wss" if secure else "ws"
        # Preserve path verbatim — some prefix-based reverse proxies (e.g. ksyun
        # web-proxy `/proxy/<port>/`) require the trailing slash for the WS
        # upgrade to match their mount; rstripping it returns the proxy's HTML
        # index page (HTTP 200) instead of upgrading.
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
                "Pi05Transport requires `websockets>=11`; pip install websockets"
            ) from exc
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        # Build kwargs, then drop any the installed websockets doesn't accept.
        # OmniGibson injects Isaac Kit's bundled (old) `websockets` into
        # sys.path on launch — its `sync.client.connect` may not know
        # `ping_interval`/`ping_timeout`/`additional_headers`/`proxy`. Probe
        # the signature and keep only supported kwargs (required ones like
        # `uri` are positional and never filtered).
        kwargs: dict = dict(
            compression=None, max_size=None,
            additional_headers=headers,
            open_timeout=self._connect_timeout_s,
            ping_interval=60, ping_timeout=300,
        )
        # `proxy=None` MUST be explicit on websockets ≥15 — without it the lib
        # auto-detects HTTP_PROXY/etc. and silently rewrites the upgrade,
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
        from . import timing as _timing  # noqa: PLC0415
        if self._ws is None:
            self._connect()
        data = self._packer.pack(dict(obs))
        with _timing.acc("infer", "n_infer"):  # VLA round-trip = inference time
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
    """openpi-compatible policy client.

    Per `policy_servers.yaml`, runs in either:
      - `shared:` mode — one server serves all 6 heads, route via stage_hint
      - legacy 6-server mode — each stage_hint pinned to its own port via
        `bound_stage_hint` for fail-fast routing checks

    Sends payload assembled by `build_policy_observation` and unwraps the
    server's response (accepts both "action" and "actions" keys).
    """

    def __init__(
        self, *, host: str, port: int, path: str = "/ws",
        bound_stage_hint: str | None = None,
        api_key: str | None = None,
        connect_timeout_s: float = 30.0,
        infer_timeout_s: float = 60.0,
        secure: bool = False,
        # Rolling inpainting (b1k-style). When inpaint is enabled, the client
        # executes only the first `actions_to_execute` of the predicted chunk
        # and saves the full chunk as the next call's `prev_actions` prior.
        # When inpaint is disabled, executes the FULL chunk (30 actions) since
        # there is no point in reserving a tail nobody will use.
        # `enable_inpaint=None` (default) auto-detects via the env var
        # `OPENPI_DISABLE_INPAINT=1` — same toggle the server uses — so the
        # client and server stay in lockstep without manual coupling.
        enable_inpaint: bool | None = None,
        actions_to_execute: int = 26,
        inpaint_overlap: int = 4,
        inpaint_until_time: float = 0.3,
    ) -> None:
        if enable_inpaint is None:
            enable_inpaint = os.environ.get("OPENPI_DISABLE_INPAINT", "0") != "1"
        self.host = host
        self.port = port
        self.path = path
        self.bound_stage_hint = bound_stage_hint
        self._transport = _Pi05Transport(
            host=host, port=port, path=path, api_key=api_key,
            connect_timeout_s=connect_timeout_s,
            infer_timeout_s=infer_timeout_s,
            secure=secure,
        )
        self._enable_inpaint = bool(enable_inpaint)
        self._actions_to_execute = int(actions_to_execute)
        self._inpaint_overlap = int(inpaint_overlap)
        self._inpaint_until_time = float(inpaint_until_time)
        # Holds the most recent `actions_raw_normalized` returned by the server,
        # which is the prior used to seed the next chunk's denoise via inpainting.
        # Cleared at episode boundaries / on subtask change via `reset_inpaint_state`.
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

        # Inpaint prior — when we have a previous chunk's raw_normalized actions,
        # send the tail (positions actions_to_execute .. +overlap) as the prior
        # for the new chunk's first `overlap` positions. The server will run
        # OT-flow-matching denoising clamping those positions until t = inpaint_until_time.
        # Server contract (per training-side patch): expects
        #   - prev_actions: shape (chunk_len, action_dim)  float32
        #   - prev_actions_mask: shape (chunk_len,)        bool
        #   - inpaint_until_time: float (optional, default 0.3)
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

        # `timeout_s` per-call overrides the transport default if provided
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
        # Always present a (T, A) chunk; if server returned (A,), wrap to (1, A)
        if actions.ndim == 1:
            actions = actions[None, :]

        # Save the full PHYSICAL chunk (pre-truncation) as the inpaint prior.
        # v11_d5 uses use_per_timestamp_norm — feeding back normalized actions
        # would be at the wrong scale (stats[26:29] vs stats[0:3]). Server now
        # accepts physical actions and renormalizes them per the new positions.
        if self._enable_inpaint and actions.ndim == 2:
            self._prev_actions_raw = actions.copy()

        # Truncate execution to `actions_to_execute` so the unexecuted tail can
        # serve as the inpaint prior for the next chunk. Only applies when
        # inpainting is enabled — otherwise execute the full chunk.
        if self._enable_inpaint and len(actions) > self._actions_to_execute:
            actions = actions[:self._actions_to_execute]

        result["actions"] = actions
        return result

    def reset_inpaint_state(self) -> None:
        """Clear the saved prev_actions prior. Call at episode start or when
        the active subtask/prompt changes — the model's previous prediction is
        no longer a reasonable prior for the new intent."""
        self._prev_actions_raw = None

    def close(self) -> None:
        self._transport.close()


class CompressionPolicyClient:
    """Champion-style action-chunk time compression wrapper. Mirrors
    `B1KPolicyWrapper._interpolate_actions` from the actual b1k src
    (`/behavior-1k-solution/src/b1k/shared/eval_b1k_wrapper.py:215-227`).

    Each `request_chunk()` call:
      1. Calls the underlying client to get the action chunk it would
         normally return (the base WebsocketPolicyClient already truncates
         to `actions_to_execute=26` when inpaint is on, and saves
         `actions[26:30]` as the inpaint prior for the next call).
      2. Resamples the chunk along the time axis from `len(actions)` to
         `execute_in_n_steps` (default 20) via per-dim cubic spline.
      3. Scales the base-velocity dims `[:, :3]` by the compression factor
         (`len/n_steps`, default 26/20 = 1.3) so the base actually moves the
         same physical distance in fewer sim steps.

    Net effect: the robot replays the same trained motion ~30% faster, and
    the policy is re-queried every `execute_in_n_steps` sim steps instead
    of every `actions_to_execute` sim steps.
    """

    def __init__(
        self,
        base: "PolicyClientProtocol",
        *,
        execute_in_n_steps: int = 20,
        velocity_dims: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        self._base = base
        self.execute_in_n_steps = int(execute_in_n_steps)
        self.velocity_dims = tuple(int(d) for d in velocity_dims)

    @property
    def server_metadata(self) -> dict | None:
        return getattr(self._base, "server_metadata", None)

    @property
    def bound_stage_hint(self) -> str | None:
        return getattr(self._base, "bound_stage_hint", None)

    def _interpolate(self, actions: np.ndarray, target_steps: int) -> np.ndarray:
        """Cubic-spline resample (T, A) → (target_steps, A). When SciPy is
        unavailable, falls back to linear via numpy.interp (still time-
        correct, just less smooth)."""
        n_in = actions.shape[0]
        if n_in == target_steps or n_in < 2:
            return actions.astype(np.float32, copy=False)
        x_in = np.linspace(0, n_in - 1, n_in, dtype=np.float64)
        x_out = np.linspace(0, n_in - 1, target_steps, dtype=np.float64)
        try:
            from scipy.interpolate import interp1d
            kind = "cubic" if n_in >= 4 else "linear"
            out = np.empty((target_steps, actions.shape[1]), dtype=np.float32)
            for d in range(actions.shape[1]):
                f = interp1d(x_in, actions[:, d], kind=kind)
                out[:, d] = f(x_out).astype(np.float32)
            return out
        except Exception:
            # Linear fallback
            out = np.empty((target_steps, actions.shape[1]), dtype=np.float32)
            for d in range(actions.shape[1]):
                out[:, d] = np.interp(x_out, x_in, actions[:, d]).astype(np.float32)
            return out

    def request_chunk(
        self, *, obs: dict[str, Any], prompt: str,
        stage_hint: str | int | None = None, timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        result = self._base.request_chunk(
            obs=obs, prompt=prompt, stage_hint=stage_hint, timeout_s=timeout_s,
        )
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] <= self.execute_in_n_steps:
            return result  # nothing to compress

        n_in = actions.shape[0]
        compressed = self._interpolate(actions, self.execute_in_n_steps)
        # Scale velocity-control dims so the same physical delta-pose happens
        # in fewer sim steps.
        compression_factor = n_in / float(self.execute_in_n_steps)
        for d in self.velocity_dims:
            if 0 <= d < compressed.shape[1]:
                compressed[:, d] *= compression_factor

        result = dict(result)
        result["actions"] = compressed
        return result

    def reset_inpaint_state(self) -> None:
        if hasattr(self._base, "reset_inpaint_state"):
            self._base.reset_inpaint_state()

    def close(self) -> None:
        if hasattr(self._base, "close"):
            self._base.close()


class MockPolicyClient:
    """In-process stub for unit tests; matches the real server response shape."""

    def __init__(self, *, stage_hint: str, chunk_size: int = 30, action_dim: int = 23) -> None:
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
    "CompressionPolicyClient",
    "MockPolicyClient",
    "build_policy_observation",
    "HEAD_RGB_KEY",
    "LEFT_RGB_KEY",
    "RIGHT_RGB_KEY",
    "PROPRIO_KEY",
    "OPENPI_HEAD_KEY",
    "OPENPI_LEFT_KEY",
    "OPENPI_RIGHT_KEY",
    "OPENPI_STATE_KEY",
]
