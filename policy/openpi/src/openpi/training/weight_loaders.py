import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class Pi05BaseToSixHeadLoader(WeightLoader):
    """Loads pi05_base into a 6-head Pi0SixHead model.

    pi05_base has the Pi0 single-expert layout:
      - PaliGemma backbone (no suffix on layer names)
      - One action expert (gemma layers with `_1` suffix)
      - One set of action_in_proj / action_out_proj / time_mlp_in / time_mlp_out

    Pi0SixHead has the 7-expert layout (PaliGemma + 6 action experts):
      - PaliGemma backbone (same names — direct copy)
      - 6 action experts (gemma layers with suffixes `_1`, `_2`, ..., `_6`)
      - 6 sets of action_in_projs[0..5] / action_out_projs[0..5] / time_mlp_ins[0..5] / time_mlp_outs[0..5]

    This loader:
      1. Direct-copies PaliGemma + SigLIP weights (matching path names).
      2. Copies the single action expert's gemma weights (`_1` suffix in
         pi05_base) into all 6 expert slots (`_1` through `_6` in the target).
         Slot `_1` is identical to source; slots `_2..._6` are renamed copies.
      3. Copies the singular IO projections (action_in_proj, action_out_proj,
         time_mlp_in, time_mlp_out) into 6 list-indexed slots each
         (action_in_projs/0, /1, ..., /5).

    All 6 experts start byte-identical; data-driven specialization (stratified
    batches) breaks the symmetry during training.
    """

    params_path: str
    num_action_experts: int = 6

    # IO layer base names that need pluralization (singular in pi05_base,
    # list-of-N in Pi0SixHead). Order: pi05 mode + non-pi05 mode covered.
    _IO_LAYER_BASES = (
        "action_in_proj",
        "action_out_proj",
        # pi05 mode:
        "time_mlp_in",
        "time_mlp_out",
        # non-pi05 mode (kept for completeness):
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    )

    # Match a name segment ending with `_1` between path separators or at end.
    # We need this to identify pi05_base's single-action-expert layer names.
    _EXPERT_1_SUFFIX_RE = re.compile(r"(?<=/)([A-Za-z][A-Za-z0-9_]*?)_1(?=/|$)")

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # nnx Python lists put int indices in pytree paths; the loader compares
        # paths as strings, so we stringify here for matching, and convert digit
        # segments back to int when unflattening at the end.
        def _flatten_str(x):
            tup = flax.traverse_util.flatten_dict(x)
            return {"/".join(str(p) for p in path): v for path, v in tup.items()}
        flat_loaded = _flatten_str(loaded)
        flat_ref = _flatten_str(params)

        result: dict[str, np.ndarray] = {}
        copied_log = {"direct": 0, "expert_replicate": 0, "io_replicate": 0, "skipped": 0}

        for k_loaded, v in flat_loaded.items():
            handled = self._try_io_replicate(k_loaded, v, flat_ref, result)
            if handled:
                copied_log["io_replicate"] += 1
                continue

            if self._is_expert_1_path(k_loaded):
                # Replicate to all 6 expert slots (slot 1 is identity rename).
                for new_eid in range(1, self.num_action_experts + 1):
                    new_k = self._rename_expert_1_to(k_loaded, new_eid)
                    if new_k in flat_ref:
                        result[new_k] = self._cast(v, flat_ref[new_k])
                copied_log["expert_replicate"] += 1
                continue

            # Direct copy if the path matches the target structure.
            if k_loaded in flat_ref:
                result[k_loaded] = self._cast(v, flat_ref[k_loaded])
                copied_log["direct"] += 1
            else:
                copied_log["skipped"] += 1

        # Fill in any param the source didn't cover (e.g. if some layer in our
        # 6-head model has no analogue in pi05_base). Logged so it's visible.
        missing = [k for k in flat_ref if k not in result]
        for k in missing:
            result[k] = flat_ref[k]
        if missing:
            logger.warning(
                "Pi05BaseToSixHeadLoader: %d target params not covered by source "
                "(kept fresh-init); first 5: %s",
                len(missing),
                missing[:5],
            )

        logger.info(
            "Pi05BaseToSixHeadLoader: copied direct=%d expert_replicate=%d io_replicate=%d "
            "(skipped %d unrecognized source keys, %d target keys kept fresh-init)",
            copied_log["direct"],
            copied_log["expert_replicate"],
            copied_log["io_replicate"],
            copied_log["skipped"],
            len(missing),
        )
        # Convert flat string-keyed dict back to nested tree, with digit-string
        # segments restored to ints so list-typed pytree nodes round-trip.
        result_tuple = {}
        for k_str, v in result.items():
            path = tuple(int(p) if p.lstrip('-').isdigit() else p for p in k_str.split("/"))
            result_tuple[path] = v
        return flax.traverse_util.unflatten_dict(result_tuple)

    # ---- helpers ----

    def _try_io_replicate(
        self,
        k_loaded: str,
        v: np.ndarray,
        flat_ref: dict[str, np.ndarray],
        result: dict[str, np.ndarray],
    ) -> bool:
        """If ``k_loaded`` is a singular IO layer (e.g. ``action_in_proj/kernel``),
        replicate into the corresponding plural list (``action_in_projs/{0..N-1}/kernel``).
        """
        for base in self._IO_LAYER_BASES:
            prefix = f"{base}/"
            if k_loaded.startswith(prefix):
                leaf = k_loaded[len(prefix) :]  # e.g. "kernel"
                plural = f"{base}s"  # e.g. "action_in_projs"
                copied_any = False
                for eid in range(self.num_action_experts):
                    new_k = f"{plural}/{eid}/{leaf}"
                    if new_k in flat_ref:
                        result[new_k] = self._cast(v, flat_ref[new_k])
                        copied_any = True
                return copied_any
        return False

    def _is_expert_1_path(self, path: str) -> bool:
        """True if ``path`` contains an `_1` suffix on a name segment inside
        the gemma hierarchy (i.e. is an action expert weight in pi05_base)."""
        # Restrict to inside the LLM hierarchy to avoid false positives.
        if "/llm/" not in path and "PaliGemma/" not in path:
            return False
        return self._EXPERT_1_SUFFIX_RE.search("/" + path) is not None

    def _rename_expert_1_to(self, path: str, new_eid: int) -> str:
        """Rename trailing `_1` segment(s) to `_<new_eid>`. Multiple `_1`
        suffixes in one path are all rewritten."""
        # Prepend "/" so the lookbehind in the pattern works on path[0].
        sentinel_path = "/" + path
        renamed = self._EXPERT_1_SUFFIX_RE.sub(rf"\1_{new_eid}", sentinel_path)
        return renamed[1:]  # strip the sentinel

    @staticmethod
    def _cast(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if hasattr(ref, "dtype") and v.dtype != ref.dtype:
            return v.astype(ref.dtype)
        return v


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    # nnx pytrees include int list indices in paths (e.g. action_in_projs/0/kernel).
    # flatten_dict(sep="/") fails on ints, so manually stringify path parts.
    def _flatten_str(x):
        tup = flax.traverse_util.flatten_dict(x)
        return {"/".join(str(p) for p in path): v for path, v in tup.items()}

    def _unflatten_str(d):
        # Inverse of _flatten_str — convert numeric segments back to int so
        # the resulting tree has the same shape as a fresh nnx.State.
        out: dict = {}
        for k, v in d.items():
            parts: list = []
            for p in k.split("/"):
                parts.append(int(p) if p.isdigit() else p)
            cur = out
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = v
        return out

    flat_ref = _flatten_str(params)
    flat_loaded = _flatten_str(loaded_params)

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return _unflatten_str(result)
