"""Regression checks for shared and legacy BEHAVIOR policy routing."""
import pytest
import yaml

from mobiagent.execution.policy_registry import PolicyRegistry
from mobiagent.execution.schemas import CANONICAL_STAGE_HINTS


SIX_STAGES = ("move_to", "pick_up_from", "place_in", "place_on", "open", "close")


@pytest.fixture(autouse=True)
def no_action_compression(monkeypatch):
    monkeypatch.delenv("CLAW_ACTION_COMPRESS", raising=False)


def config_file(tmp_path, config):
    path = tmp_path / "servers.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def legacy_config():
    return {stage: {"host": "127.0.0.1", "port": 8000 + i}
            for i, stage in enumerate(SIX_STAGES)}


def test_legacy_six_servers_do_not_require_alternative_heads(tmp_path):
    registry = PolicyRegistry.from_yaml(config_file(tmp_path, legacy_config()))
    try:
        clients = [registry.select(stage) for stage in SIX_STAGES]
        assert len({id(client) for client in clients}) == 6
        assert [client.bound_stage_hint for client in clients] == list(SIX_STAGES)
        with pytest.raises(KeyError, match="unknown stage_hint"):
            registry.select("place")
    finally:
        registry.close()


def test_optional_legacy_alias_has_its_own_bound_client(tmp_path):
    config = legacy_config()
    config["place"] = {"port": 8100}
    registry = PolicyRegistry.from_yaml(config_file(tmp_path, config))
    try:
        assert registry.select("place").bound_stage_hint == "place"
        assert registry.select("place") is not registry.select("place_in")
    finally:
        registry.close()


def test_incomplete_legacy_layout_still_fails(tmp_path):
    config = legacy_config()
    del config["close"]
    with pytest.raises(ValueError, match="missing stage_hint 'close'"):
        PolicyRegistry.from_yaml(config_file(tmp_path, config))


def test_direct_registry_still_requires_six_experts():
    with pytest.raises(ValueError, match="missing clients"):
        PolicyRegistry({})


def test_shared_registry_retains_all_checkpoint_routes(tmp_path):
    registry = PolicyRegistry.from_yaml(config_file(tmp_path, {"shared": {"port": 8000}}))
    try:
        assert registry.is_shared
        clients = [registry.select(stage) for stage in CANONICAL_STAGE_HINTS]
        assert len({id(client) for client in clients}) == 1
        assert clients[0].bound_stage_hint is None
    finally:
        registry.close()
