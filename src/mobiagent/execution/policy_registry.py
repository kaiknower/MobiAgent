"""Stage-hint -> policy-client routing layer.

Two YAML deployment shapes:

1) **shared** (default — matches our shared backbone + 6-head training):
       shared:
         host: localhost
         port: 8001
         path: /ws
   ONE WebsocketPolicyClient instance is reused across all 6 stage_hints.
   The request payload includes `stage_hint` so the server routes to the
   correct head internally.

2) **legacy 6-server** (one ckpt per stage_hint, e.g. stage-specialized
   fine-tuning where each head was trained independently):
       move_to:       {host: localhost, port: 8001}
       pick_up_from:  {host: localhost, port: 8002}
       ...
   One client per stage_hint, each `bound_stage_hint`-pinned for fail-fast
   integrity checks.

Mock mode (`PolicyRegistry.mock()`): one MockPolicyClient per stage_hint so
tests can assert routing keys exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import os

from .policy_client import (
    CompressionPolicyClient,
    MockPolicyClient,
    PolicyClientProtocol,
    WebsocketPolicyClient,
)


def _maybe_wrap_compression(client: PolicyClientProtocol) -> PolicyClientProtocol:
    """Optionally wrap a base client with CompressionPolicyClient — champion's
    cubic-spline action compression (`/behavior-1k-solution/src/b1k/shared/
    eval_b1k_wrapper.py`). Enabled when `CLAW_ACTION_COMPRESS=1`. Tunables:
      - CLAW_EXECUTE_STEPS  (default 20)  — target sim-step count per chunk
      - CLAW_VELOCITY_DIMS  (default "0,1,2") — comma-separated indices to
        scale by `len_in / execute_steps` (base velocity channels).
    """
    if os.environ.get("CLAW_ACTION_COMPRESS", "0") != "1":
        return client
    vd_raw = os.environ.get("CLAW_VELOCITY_DIMS", "0,1,2")
    velocity_dims = tuple(int(x) for x in vd_raw.split(",") if x.strip())
    return CompressionPolicyClient(
        client,
        execute_in_n_steps=int(os.environ.get("CLAW_EXECUTE_STEPS", "20")),
        velocity_dims=velocity_dims,
    )
from .schemas import CANONICAL_STAGE_HINTS


class PolicyRegistry:
    def __init__(self, clients: dict[str, PolicyClientProtocol], *, shared_client: PolicyClientProtocol | None = None) -> None:
        # When shared_client is set, all stage_hints route to the same client.
        # `clients` is still populated (each key → the shared instance) so the
        # rest of the code path is uniform.
        missing = [h for h in CANONICAL_STAGE_HINTS if h not in clients]
        if missing:
            raise ValueError(f"PolicyRegistry missing clients for stage_hints: {missing}")
        self._clients = clients
        self._shared_client = shared_client

    @property
    def is_shared(self) -> bool:
        return self._shared_client is not None

    def select(self, stage_hint: str) -> PolicyClientProtocol:
        if stage_hint not in self._clients:
            raise KeyError(f"unknown stage_hint: {stage_hint!r}; valid={CANONICAL_STAGE_HINTS}")
        return self._clients[stage_hint]

    def close(self) -> None:
        seen: set[int] = set()
        for c in self._clients.values():
            if id(c) in seen:
                continue
            seen.add(id(c))
            try:
                c.close()
            except Exception:
                pass

    def reset_inpaint_state(self) -> None:
        """Reset rolling-inpaint prior on every unique underlying client.
        Idempotent on clients that don't support it."""
        seen: set[int] = set()
        for c in self._clients.values():
            if id(c) in seen:
                continue
            seen.add(id(c))
            fn = getattr(c, "reset_inpaint_state", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PolicyRegistry":
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PolicyRegistry.from_yaml requires PyYAML") from exc
        cfg = yaml.safe_load(Path(config_path).read_text())

        # Mode 1: shared — single client serves all 6 stage_hints
        if "shared" in cfg and cfg["shared"]:
            entry = cfg["shared"]
            shared_base = WebsocketPolicyClient(
                host=entry.get("host", "localhost"),
                port=int(entry["port"]),
                path=entry.get("path", "/ws"),
                bound_stage_hint=None,  # not bound — accepts any stage_hint
                secure=bool(entry.get("secure", False)),
            )
            shared = _maybe_wrap_compression(shared_base)
            clients = {h: shared for h in CANONICAL_STAGE_HINTS}
            return cls(clients, shared_client=shared)

        # Mode 2: legacy 6-server (one ckpt per stage_hint)
        clients: dict[str, PolicyClientProtocol] = {}
        for h in CANONICAL_STAGE_HINTS:
            entry = cfg.get(h)
            if not entry:
                raise ValueError(f"{config_path}: missing stage_hint {h!r} (and no `shared:` block)")
            base = WebsocketPolicyClient(
                host=entry.get("host", "localhost"),
                port=int(entry["port"]),
                path=entry.get("path", "/ws"),
                bound_stage_hint=h,
                secure=bool(entry.get("secure", False)),
            )
            clients[h] = _maybe_wrap_compression(base)
        return cls(clients)

    @classmethod
    def mock(cls, *, chunk_size: int = 30, action_dim: int = 22) -> "PolicyRegistry":
        clients = {
            h: MockPolicyClient(stage_hint=h, chunk_size=chunk_size, action_dim=action_dim)
            for h in CANONICAL_STAGE_HINTS
        }
        return cls(clients)


__all__ = ["PolicyRegistry"]
