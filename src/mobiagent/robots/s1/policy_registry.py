"""Stage-hint → policy-client routing for the 3-head S1-mobile policy.

Two YAML deployment shapes (`configs/policy_servers.yaml.example`):

1) **shared** (recommended — one server hosts all 3 heads):
       shared:
         host: localhost
         port: 8001
         path: /ws
   ONE WebsocketPolicyClient is reused across all stage_hints. The
   request payload includes `stage_override` so the server routes to
   the correct head internally.

2) **per-head 3-server** (one ckpt per stage_hint):
       move_to: {host: localhost, port: 8001}
       pick_up: {host: localhost, port: 8002}
       place:   {host: localhost, port: 8003}
   One client per stage_hint, each `bound_stage_hint`-pinned for fail-
   fast integrity checks.

Mock mode (`PolicyRegistry.mock()`): one MockPolicyClient per
stage_hint so tests can assert routing keys exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy_client import (
    MockPolicyClient,
    PolicyClientProtocol,
    WebsocketPolicyClient,
)
from .schemas import CANONICAL_STAGE_HINTS


class PolicyRegistry:
    def __init__(
        self,
        clients: dict[str, PolicyClientProtocol],
        *,
        shared_client: PolicyClientProtocol | None = None,
    ) -> None:
        missing = [h for h in CANONICAL_STAGE_HINTS if h not in clients]
        if missing:
            raise ValueError(f"PolicyRegistry missing clients for stage_hints: {missing}")
        self._clients = clients
        self._shared_client = shared_client

    @property
    def is_shared(self) -> bool:
        return self._shared_client is not None

    def select(self, stage_hint: str) -> PolicyClientProtocol:
        # Planner-side synonyms collapse to canonical heads.
        canonical = {
            "move_to":      "move_to",
            "pick_up":      "pick_up",
            "pick_up_from": "pick_up",
            "place":        "place",
            "place_in":     "place",
            "place_on":     "place",
            "pour_into":    "place",
        }.get(stage_hint, stage_hint)
        if canonical not in self._clients:
            raise KeyError(
                f"unknown stage_hint: {stage_hint!r}; canonical heads={CANONICAL_STAGE_HINTS}"
            )
        return self._clients[canonical]

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

        # Mode 1: shared — single client serves all 3 heads
        if "shared" in cfg and cfg["shared"]:
            entry = cfg["shared"]
            shared = WebsocketPolicyClient(
                host=entry.get("host", "localhost"),
                port=int(entry["port"]),
                path=entry.get("path", "/ws"),
                bound_stage_hint=None,
                secure=bool(entry.get("secure", False)),
            )
            clients = {h: shared for h in CANONICAL_STAGE_HINTS}
            return cls(clients, shared_client=shared)

        # Mode 2: per-head 3-server
        clients: dict[str, PolicyClientProtocol] = {}
        for h in CANONICAL_STAGE_HINTS:
            entry = cfg.get(h)
            if not entry:
                raise ValueError(
                    f"{config_path}: missing stage_hint {h!r} (and no `shared:` block)"
                )
            clients[h] = WebsocketPolicyClient(
                host=entry.get("host", "localhost"),
                port=int(entry["port"]),
                path=entry.get("path", "/ws"),
                bound_stage_hint=h,
                secure=bool(entry.get("secure", False)),
            )
        return cls(clients)

    @classmethod
    def mock(cls, *, chunk_size: int = 30, action_dim: int = 34) -> "PolicyRegistry":
        clients = {
            h: MockPolicyClient(stage_hint=h, chunk_size=chunk_size, action_dim=action_dim)
            for h in CANONICAL_STAGE_HINTS
        }
        return cls(clients)


__all__ = ["PolicyRegistry"]
