import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.pi0_six_head_config as pi0_six_head_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.behavior_policy as behavior_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """
    assets_dir: str | None = None
    asset_id: str | None = None

@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    per_expert_norm_stats: list[dict[str, _transforms.NormStats]] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    use_per_timestamp_norm: bool = False
    use_per_expert_norm: bool = False
    action_sequence_keys: Sequence[str] = ('actions',)
    prompt_from_task: bool = False
    rlds_data_dir: str | None = None
    behavior_dataset_root: str | None = None
    behavior_manifest_path: str | None = None
    behavior_frame_cache_root: str | None = None
    behavior_video_tolerance_s: float = 0.2
    behavior_runtime_stage_id: int | None = None
    behavior_task_index_filter: int | None = None
    behavior_prompt_style: Literal['canonical', 'task0_runtime'] = 'canonical'
    behavior_stage_balanced_sampling: bool = False
    skill_segments_dir: str | None = None
    skill_segments_sampler_weights_path: str | None = None
    skill_segments_per_expert: int = 4
    skill_segments_num_experts: int = 6
    skill_segments_random_window: bool = True
    skill_segments_per_expert_proportional: bool = True
    skill_segments_per_expert_chunk_proportional: bool = False
    skill_segments_per_expert_counts_override: tuple[int, ...] | None = None
    skill_segments_chunk_weighted: bool = False
    skill_segments_task_balanced_within_head: bool = False
    skill_segments_use_per_frame_state_delta: bool = False
    skill_segments_single_head_index: int | None = None
    skill_segments_canonical_heads: tuple[str, ...] | None = None
    skill_segments_prompt_style: str = 'task_then_now'
    skill_segments_state_column: str = 'observation.state'
    skill_segments_action_column: str = 'action'
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()

class GroupFactory(Protocol):

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len)), _transforms.PadStatesAndActions(model_config.action_dim)])
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer(model_config.max_token_len), discrete_state_input=model_config.discrete_state_input), _transforms.PadStatesAndActions(model_config.action_dim)])
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = _tokenizer.FASTTokenizer if model_config.fast_model_tokenizer is None else model_config.fast_model_tokenizer
                tokenizer_kwargs = {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                return _transforms.Group(inputs=[_transforms.InjectDefaultPrompt(self.default_prompt), _transforms.ResizeImages(224, 224), _transforms.TokenizeFASTInputs(tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs))], outputs=[_transforms.ExtractFASTActions(tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs), action_horizon=model_config.action_horizon, action_dim=model_config.action_dim)])

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(self.base_config or DataConfig(), repo_id=repo_id, asset_id=asset_id, norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id), use_quantile_norm=model_config.model_type != ModelType.PI0)

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f'Loaded norm stats from {data_assets_dir}')
            return norm_stats
        except FileNotFoundError:
            logging.info(f'Norm stats not found in {data_assets_dir}, skipping.')
        return None

@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = 'fake'

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)

@dataclasses.dataclass(frozen=True)
class BehaviorSegmentDataConfig(DataConfigFactory):
    behavior_dataset_root: str = tyro.MISSING
    manifest_path: str = tyro.MISSING
    frame_cache_root: str | None = None
    video_tolerance_s: float = 0.2
    runtime_stage_id: int | None = None
    task_index_filter: int | None = None
    prompt_style: Literal['canonical', 'task0_runtime'] = 'canonical'
    stage_balanced_sampling: bool = False
    use_per_timestamp_norm: bool = False
    force_use_quantile_norm: bool | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(inputs=[behavior_policy.BehaviorInputs(model_type=model_config.model_type)], outputs=[behavior_policy.BehaviorOutputs()])
        model_transforms = ModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(base_config, data_transforms=data_transforms, model_transforms=model_transforms, action_sequence_keys=('action',), use_quantile_norm=base_config.use_quantile_norm if self.force_use_quantile_norm is None else self.force_use_quantile_norm, use_per_timestamp_norm=self.use_per_timestamp_norm, behavior_dataset_root=self.behavior_dataset_root, behavior_manifest_path=self.manifest_path, behavior_frame_cache_root=self.frame_cache_root, behavior_video_tolerance_s=self.video_tolerance_s, behavior_runtime_stage_id=self.runtime_stage_id, behavior_task_index_filter=self.task_index_filter, behavior_prompt_style=self.prompt_style, behavior_stage_balanced_sampling=self.stage_balanced_sampling)

@dataclasses.dataclass(frozen=True)
class SkillSegmentsDataConfig(DataConfigFactory):
    """Data configuration for routed skill-segment training.

    Reads pre-sliced ``head__*.jsonl`` shards (built by ``scripts/data/split_per_head.py``). Pairs with :class:`StratifiedWeightedBatchSampler`
    to feed Pi0SixHead with stratified batches.

    Action representation:
    - dims 0-2 (base velocity): **absolute** (already a velocity)
    - dims 3-5 (trunk first 3 joints): **delta** from current state
    - dim 6  (trunk 4th joint, e.g. trunk lift): **absolute**
    - dims 7-13 (left arm 7 joints): **delta**
    - dim 14 (left gripper width): **absolute**
    - dims 15-21 (right arm 7 joints): **delta**
    - dim 22 (right gripper width): **absolute**
    """
    skill_segments_dir: str = tyro.MISSING
    sampler_weights_path: str | None = None
    per_expert: int = 4
    num_experts: int = 6
    random_window: bool = True
    use_per_timestamp_norm: bool = False
    use_per_expert_norm: bool = False
    force_use_quantile_norm: bool | None = None
    use_delta_joint_actions: bool = True
    use_per_frame_state_delta: bool = False
    per_expert_proportional: bool = True
    per_expert_chunk_proportional: bool = False
    per_expert_counts_override: tuple[int, ...] | None = None
    chunk_weighted: bool = False
    task_balanced_within_head: bool = False
    single_head_index: int | None = None
    canonical_heads: tuple[str, ...] | None = None
    skill_prompt_style: str = 'task_then_now'
    state_column: str = 'observation.state'
    action_column: str = 'action'

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        from openpi.policies import skill_segment_policy
        delta_action_mask = (
            _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
            if self.use_delta_joint_actions else _transforms.make_bool_mask(-23)
        )
        data_transforms = _transforms.Group(
            inputs=[
                skill_segment_policy.SkillSegmentInputs(model_type=model_config.model_type),
                _transforms.DeltaActions(delta_action_mask, state_key='action_delta_state' if self.use_per_frame_state_delta else 'state'),
                _transforms.DropKeys(('action_delta_state',)),
            ],
            outputs=[
                _transforms.RollingAbsoluteActions(delta_action_mask) if self.use_per_frame_state_delta else _transforms.AbsoluteActions(delta_action_mask),
                skill_segment_policy.SkillSegmentOutputs(),
            ],
        )
        model_transforms = ModelTransformFactory()(model_config)
        base_config = self.create_base_config(assets_dirs, model_config)
        sampler_weights_path = self.sampler_weights_path
        if sampler_weights_path is None:
            sampler_weights_path = str(pathlib.Path(self.skill_segments_dir) / 'sampler_weights.json')
        per_expert_norm_stats = self._load_per_expert_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), base_config.asset_id, canonical_heads=self.canonical_heads)
        return dataclasses.replace(base_config, data_transforms=data_transforms, model_transforms=model_transforms, action_sequence_keys=('action',), use_quantile_norm=base_config.use_quantile_norm if self.force_use_quantile_norm is None else self.force_use_quantile_norm, use_per_timestamp_norm=self.use_per_timestamp_norm, use_per_expert_norm=self.use_per_expert_norm, per_expert_norm_stats=per_expert_norm_stats, skill_segments_dir=self.skill_segments_dir, skill_segments_sampler_weights_path=sampler_weights_path, skill_segments_per_expert=self.per_expert, skill_segments_num_experts=self.num_experts, skill_segments_random_window=self.random_window, skill_segments_per_expert_proportional=self.per_expert_proportional, skill_segments_per_expert_chunk_proportional=self.per_expert_chunk_proportional, skill_segments_per_expert_counts_override=self.per_expert_counts_override, skill_segments_chunk_weighted=self.chunk_weighted, skill_segments_task_balanced_within_head=self.task_balanced_within_head, skill_segments_use_per_frame_state_delta=self.use_per_frame_state_delta, skill_segments_single_head_index=self.single_head_index, skill_segments_canonical_heads=self.canonical_heads, skill_segments_prompt_style=self.skill_prompt_style, skill_segments_state_column=self.state_column, skill_segments_action_column=self.action_column)

    @staticmethod
    def _load_per_expert_norm_stats(assets_dir: epath.Path, asset_id: str | None, canonical_heads: tuple[str, ...] | None=None) -> list[dict[str, _transforms.NormStats]] | None:
        from openpi.training.skill_segment_dataset import CANONICAL_HEADS
        if asset_id is None:
            return None
        heads = canonical_heads if canonical_heads is not None else CANONICAL_HEADS
        per_expert_root = pathlib.Path(str(assets_dir / asset_id)) / 'per_expert'
        if not per_expert_root.exists():
            logging.info(f'Per-expert norm stats not found at {per_expert_root}, falling back to combined stats.')
            return None
        out: list[dict[str, _transforms.NormStats]] = []
        for name in heads:
            head_dir = per_expert_root / name
            try:
                stats = _normalize.load(_download.maybe_download(str(head_dir)))
            except FileNotFoundError:
                logging.warning(f'Missing per-expert stats {head_dir} — disabling PerExpertNormalize.')
                return None
            out.append(stats)
        logging.info(f'Loaded {len(out)} per-expert norm stats from {per_expert_root}')
        return out

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = 'openpi'
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal['bfloat16', 'float32'] = 'bfloat16'
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = './assets'
    checkpoint_base_dir: str = './checkpoints'
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30000
    grad_accum_steps: int = 1
    per_head_loss_weight: float = 0.0
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError('--exp_name must be set')
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError('Cannot resume and overwrite at the same time.')

def _six_head_freeze_paligemma_backbone_filter() -> Filter:
    """Freeze the vision and language backbone for BEHAVIOR expert training.

    Action expert towers, their input/output projections and the subtask head
    remain trainable."""
    return nnx.Any(nnx_utils.PathRegex('.*img.*'), nnx.All(nnx_utils.PathRegex('.*llm.*'), nnx.Not(nnx_utils.PathRegex('.*llm.*_[1-6].*'))))

# Portable baseline recipes; dataset and checkpoint locations are user supplied.
import os

def _skill_recipe(name, heads):
    return TrainConfig(
        name=name, exp_name="run", project_name="mobiagent",
        model=pi0_six_head_config.Pi0SixHeadConfig(
            expert_names=heads, action_dim=32,
            action_horizon=50, max_token_len=200,
            discrete_state_input=True),
        freeze_filter=_six_head_freeze_paligemma_backbone_filter(),
        weight_loader=weight_loaders.Pi05BaseToSixHeadLoader(
            params_path=os.environ.get("MOBIAGENT_BASE_PARAMS", "gs://openpi-assets/checkpoints/pi05_base/params"),
            num_action_experts=len(heads)),
        data=SkillSegmentsDataConfig(
            repo_id=name,
            skill_segments_dir=os.environ.get("MOBIAGENT_SEGMENTS_DIR", "data/segments"),
            num_experts=len(heads), canonical_heads=heads,
            per_expert=4, per_expert_proportional=False,
            use_per_expert_norm=True, force_use_quantile_norm=False,
            state_column="observation.state",
            action_column="action",
            skill_prompt_style="skill_only"),
        batch_size=4*len(heads), num_workers=2, fsdp_devices=1,
        wandb_enabled=False,
        assets_base_dir=os.environ.get("MOBIAGENT_ASSETS_DIR", "assets"),
        checkpoint_base_dir=os.environ.get("MOBIAGENT_CHECKPOINT_DIR", "checkpoints"))

_CONFIGS = [
    _skill_recipe("mobiagent_behavior", ("move_to", "pick_up_from", "place_in", "place_on", "open", "close")),
    TrainConfig(name="debug", exp_name="debug", model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"), data=FakeDataConfig(), batch_size=2, num_train_steps=2, wandb_enabled=False),
]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}

def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def get_config(config_name: str) -> TrainConfig:
    if config_name == "mobiagent_robocasa":
        from openpi.training.robocasa_data import RoboCasaV3Data, model_config
        return TrainConfig(name="mobiagent_robocasa", exp_name="run", model=model_config(), data=RoboCasaV3Data(), wandb_enabled=False)
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ''
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
