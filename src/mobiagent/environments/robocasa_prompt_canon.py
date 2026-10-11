from __future__ import annotations
from pathlib import Path
import functools
import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
DEFAULT_INSTR_DIR = str(Path(__file__).with_name("robocasa_instructions"))
_WS = re.compile("\\s+")
_PUNCT = re.compile("[^a-z0-9 ]")
_ORDINAL = re.compile("\\s*\\(\\s*\\d+\\s*/\\s*\\d+\\s*\\)\\s*$")


def _norm(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", _ORDINAL.sub("", str(s)).lower())).strip()


@functools.lru_cache(maxsize=1)
def _vocab() -> tuple[dict[str, str], dict[str, set[str]]]:
    instr_dir = os.environ.get("CLAW_INSTR_DIR", DEFAULT_INSTR_DIR)
    by_norm: dict[str, str] = {}
    skills_of: dict[str, set[str]] = {}
    for path in sorted(glob.glob(os.path.join(instr_dir, "*.json"))):
        try:
            with open(path) as f:
                d = json.load(f)
            if not isinstance(d, dict) or not isinstance(d.get("instructions"), dict):
                continue
            skill = d["skill"]
            for s in d["instructions"]:
                if " " not in s:
                    continue
                by_norm.setdefault(_norm(s), s)
                skills_of.setdefault(s, set()).add(skill)
        except Exception:
            logger.exception("[canon] failed to load %s", path)
    if not by_norm:
        logger.warning(
            "[canon] no instruction vocab under %s — exact matching disabled, template rules only",
            instr_dir,
        )
    return (by_norm, skills_of)


def _cap(s: str) -> str:
    s = s.strip().rstrip(".") + "."
    return s[0].upper() + s[1:] if s else s


_BURNER_POS = re.compile(
    "\\b(front|rear|back)\\b.{0,12}?\\b(left|right|center)\\b|\\b(left|right|center)\\b.{0,12}?\\b(front|rear|back)\\b|\\b(center)\\b"
)
_ON_WORDS = re.compile("\\b(turn on|activate|ignite|light|switch on|start)\\b")
_OFF_WORDS = re.compile("\\b(turn off|deactivate|switch off|extinguish|stop)\\b")
_ART = "(?:(?:the|a|an|either) )?"
_PNP = re.compile(
    "\\bpick(?: up)? "
    + _ART
    + "(?P<obj>.+?) (?:from|out of|off) "
    + _ART
    + "(?P<src>.+?) and (?:place|put|set) (?:it|them) (?P<prep>in|into|on|onto|under) "
    + _ART
    + "(?P<dst>.+)$"
)
_PICK_ONLY = re.compile(
    "\\bpick(?: up)? "
    + _ART
    + "(?P<obj>.+?)(?: (?:from|out of|off) "
    + _ART
    + "(?P<src>.+))?$"
)
_PLACE_ONLY = re.compile(
    "\\b(?:place|put|set) "
    + _ART
    + "(?P<obj>.+?) (?P<prep>in|into|on|onto|under) "
    + _ART
    + "(?P<dst>.+)$"
)
_NAV = re.compile(
    "\\b(?:navigate|go|move|walk|drive) (?:back )?to " + _ART + "(?P<dst>.+)$"
)
_OPENCLOSE = re.compile(
    "\\b(?:(?P<open>open|pull open|slide open|pull out)|(?P<close>close|shut|slide in|push in|push shut)) (?:the )?(?P<what>.+?)(?: (?:door|doors))?$"
)


def _burner_pos(t: str) -> str | None:
    m = _BURNER_POS.search(t)
    if not m:
        return None
    g = [x for x in m.groups() if x]
    g = ["rear" if x == "back" else x for x in g]
    if g == ["center"]:
        return "center"
    fr = next((x for x in g if x in ("front", "rear")), None)
    lr = next((x for x in g if x in ("left", "right", "center")), None)
    if fr and lr:
        return f"{fr} {lr}"
    return None


def _rule_switch(t: str) -> str | None:
    on = bool(_ON_WORDS.search(t)) or "press" in t or "push" in t
    off = bool(_OFF_WORDS.search(t)) or t.startswith("close ")
    if "burner" in t or "stove" in t or ("knob" in t and "oven" not in t):
        pos = _burner_pos(t)
        if pos:
            verb = "Turn off" if off and (not _ON_WORDS.search(t)) else "Turn on"
            return f"{verb} the {pos} burner of the stove."
        return None
    if "spout" in t and "wash" in t:
        return "Move the spout to wash all locations of the sink basin."
    if "faucet" in t or ("sink" in t and ("water" in t or on or off)):
        return "Turn off the sink faucet." if off else "Turn on the sink faucet."
    if "microwave" in t:
        return (
            "Press the stop button on the microwave."
            if off
            else "Press the start button on the microwave."
        )
    if "kettle" in t:
        return "Press down the lever to turn on the electric kettle."
    if "toaster oven" in t:
        if "increase" in t:
            return "Increase the toaster oven temperature."
        if "decrease" in t or "lower" in t:
            return "Decrease the toaster oven temperature."
        return "Turn on the toaster oven by setting the timer."
    if "toaster" in t:
        return "Push down the lever of the toaster to turn it on."
    if "blender" in t:
        return "Turn on the blender by pressing the power button."
    if "coffee" in t:
        return "Press the button on the coffee machine to serve coffee."
    if "oven" in t and ("preheat" in t or "temperature" in t or "knob" in t):
        return "Preheat the oven by turning the temperature knob."
    return None


_FIXTURE_DOOR = {
    "fridge": "fridge door",
    "microwave": "microwave door",
    "oven": "oven door",
    "toaster oven": "toaster oven door",
    "dishwasher": "dishwasher door",
}


def _rule_openclose(t: str, closing: bool) -> str | None:
    verb = "Close" if closing else "Open"
    if "fridge drawer" in t or "refrigerator drawer" in t:
        return f"Fully {verb.lower()} the fridge drawer."
    if "dishwasher rack" in t:
        return f"Fully slide the top dishwasher rack {('in' if closing else 'out')}."
    if "toaster oven" in t and ("rack" in t or "tray" in t):
        return f"Fully slide the toaster oven rack {('in' if closing else 'out')}."
    if "oven rack" in t:
        return f"Fully slide the oven rack {('in' if closing else 'out')}."
    if "kettle" in t or ("lid" in t and "blender" not in t and ("kettle" in t)):
        return (
            "Close the lid of the electric kettle."
            if closing
            else "Press the button to open the lid of the electric kettle."
        )
    if "blender" in t:
        return (
            "Close the lid blender by securely placing the lid on top."
            if closing
            else "Open the blender by taking off the lid and placing it on the counter."
        )
    if "stand mixer" in t or "mixer head" in t:
        return f"{verb} the stand mixer head."
    if "cabinet" in t:
        n = "doors" if "doors" in t else "door"
        return f"{verb} the cabinet {n}."
    if "drawer" in t:
        for side in ("left", "right"):
            if side in t:
                return f"{verb} the {side} drawer."
        return f"{verb} the drawer."
    for fx in ("toaster oven", "fridge", "microwave", "dishwasher", "oven"):
        if fx in t:
            n = _FIXTURE_DOOR[fx]
            if fx == "fridge" and "doors" in t:
                n = "fridge doors"
            return f"{verb} the {n}."
    return None


_LATERAL = ("left", "right")


def _p7_strip_lateral(
    norm_s: str, by_norm: dict[str, str], goal_norm: str | None
) -> str | None:
    toks = norm_s.split()
    for i, tok in enumerate(toks):
        if tok not in _LATERAL:
            continue
        if goal_norm and re.search(f"\\b{tok}\\b", goal_norm):
            continue
        hit = by_norm.get(" ".join(toks[:i] + toks[i + 1 :]))
        if hit is not None:
            return hit
    return None


def _p8_strip_of_suffix(norm_s: str, by_norm: dict[str, str]) -> str | None:
    idx = norm_s.rfind(" of the ")
    if idx <= 0:
        return None
    if not norm_s[idx + len(" of the ") :].strip():
        return None
    return by_norm.get(norm_s[:idx].rstrip())


def _p9_snap_redundant_appliance_detail(
    norm_s: str, by_norm: dict[str, str]
) -> str | None:
    candidates: list[str] = []
    if "electric kettle" in norm_s:
        candidates.append(norm_s.replace("electric kettle", "kettle"))
    if norm_s.startswith("navigate to "):
        for suffix in (
            " on the dining counter",
            " at the dining counter",
            " on the counter",
            " at the counter",
        ):
            if norm_s.endswith(suffix):
                candidates.append(norm_s[: -len(suffix)].rstrip())
                candidates.append("navigate to the counter")
    return next(
        (by_norm[candidate] for candidate in candidates if candidate in by_norm), None
    )


def _p10_insert_missing_definite_article(
    norm_s: str, by_norm: dict[str, str]
) -> str | None:
    tokens = norm_s.split()
    if len(tokens) < 2 or tokens[1] == "the":
        return None
    candidate = " ".join((tokens[0], "the", *tokens[1:]))
    return by_norm.get(candidate)


def _p11_snap_kettle_stove_burner(norm_s: str, by_norm: dict[str, str]) -> str | None:
    if re.fullmatch(
        "(?:place|put|set) (?:the )?kettle (?:on|onto) (?:the )?(?:(?:front|rear|back|left|right|center) ){1,2}burner of (?:the )?stove",
        norm_s,
    ):
        return by_norm.get("place the kettle on the stove burner")
    return None


def _p12_snap_sink_action(norm_s: str, by_norm: dict[str, str]) -> str | None:
    if re.fullmatch("(?:open|start|turn on) (?:the )?sink", norm_s):
        return by_norm.get("turn on the sink faucet")
    if re.fullmatch("(?:wash|rinse) (?:the )?sink basin", norm_s):
        return by_norm.get("move the spout to wash all locations of the sink basin")
    return None


def _p13_strip_redundant_place_support(
    norm_s: str, by_norm: dict[str, str]
) -> str | None:
    if not norm_s.startswith(("place ", "put ", "set ")):
        return None
    for suffix in (
        " on the nearby counter",
        " at the nearby counter",
        " on nearby counter",
        " at nearby counter",
        " on the dining counter",
        " at the dining counter",
        " on dining counter",
        " at dining counter",
        " on the kitchen counter",
        " at the kitchen counter",
        " on kitchen counter",
        " at kitchen counter",
        " on the counter",
        " at the counter",
        " on counter",
        " at counter",
    ):
        if norm_s.endswith(suffix):
            candidate = norm_s[: -len(suffix)].rstrip()
            hit = by_norm.get(candidate)
            if hit is not None:
                return hit
    return None


_NO_ACTION = re.compile("\\b(wait|waiting|hold still|observe|watch|check if)\\b")
_ACTION_START = re.compile(
    "^(pick|place|put|set|open|close|shut|turn|press|push|pull|navigate|move|go|walk|stack|pour|slide|grab|grasp|lift|insert|remove|take|twist|flip|shake|stir|scrub|wipe|rinse|wash|drop|release|adjust|preheat|lower|increase|decrease|fully|start|stop|activate)\\b"
)
_SWITCH_INTENT = re.compile(
    "\\b(press|push|turn|twist|switch|start|stop|activate|deactivate|ignite|open|close|knob|lever|button|faucet|spout|preheat|increase|decrease|adjust|dial|timer)\\b"
)


def _apply_rules(t: str, stage_hint) -> str | None:
    hint = str(stage_hint or "")
    if _NO_ACTION.search(t):
        return None
    m = _PNP.search(t)
    if m:
        prep = {"into": "in", "onto": "on"}.get(m["prep"], m["prep"])
        return f"Pick the {m['obj']} from the {m['src']} and place it {prep} the {m['dst']}."
    if " and place" not in t and " and put" not in t:
        m = _PICK_ONLY.search(t)
        if m:
            if m["src"]:
                return f"Pick up the {m['obj']} from the {m['src']}."
            return f"Pick up the {m['obj']}."
        m = _PLACE_ONLY.search(t)
        if m:
            prep = {"into": "in", "onto": "on"}.get(m["prep"], m["prep"])
            return f"Place the {m['obj']} {prep} the {m['dst']}."
    m = _NAV.search(t)
    if m and ("pick" not in t and "place" not in t):
        return f"Navigate to the {m['dst']}."
    m = re.search("\\bslide\\b.*\\brack\\b.*\\b(in|out)\\b", t)
    if m:
        r = _rule_openclose(t, closing=m.group(1) == "in")
        if r:
            return r
    if _SWITCH_INTENT.search(t):
        sw = _rule_switch(t)
        if sw:
            return sw
    if re.search("\\b(open|pull out|slide out|pull open)\\b", t) and hint != "close":
        r = _rule_openclose(t, closing=False)
        if r:
            return r
    if re.search("\\b(close|shut|slide in|push in)\\b", t) or hint == "close":
        r = _rule_openclose(t, closing=True)
        if r:
            return r
    return None


_seen: set[str] = set()


def canonicalize(prompt: str, stage_hint=None, goal=None) -> tuple[str, set[str]]:
    raw = str(prompt or "").strip()
    by_norm, skills_of = _vocab()
    t = _norm(raw).replace("refrigerator", "fridge")
    out = by_norm.get(t)
    if out is None:
        ruled = _apply_rules(t, stage_hint)
        if ruled is not None:
            out = by_norm.get(_norm(ruled), ruled)
    if out is None or out not in skills_of:
        goal_norm = _norm(goal) if goal else None
        cands = [t]
        if out is not None and _norm(out) != t:
            cands.append(_norm(out))
        for c in cands:
            hit = _p7_strip_lateral(c, by_norm, goal_norm)
            if hit is not None:
                logger.info(
                    "[canon-p7] %r -> %r (ungrounded lateral qualifier stripped)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p8_strip_of_suffix(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p8] %r -> %r (redundant trailing 'of the ...' stripped)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p9_snap_redundant_appliance_detail(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p9] %r -> %r (redundant appliance detail stripped)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p10_insert_missing_definite_article(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p10] %r -> %r (missing definite article restored onto exact vocab hit)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p11_snap_kettle_stove_burner(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p11] %r -> %r (ungrounded kettle burner position stripped onto exact vocab hit)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p12_snap_sink_action(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p12] %r -> %r (terse sink action snapped onto exact vocab hit)",
                    raw,
                    hit,
                )
                out = hit
                break
            hit = _p13_strip_redundant_place_support(c, by_norm)
            if hit is not None:
                logger.info(
                    "[canon-p13] %r -> %r (redundant place support stripped onto exact vocab hit)",
                    raw,
                    hit,
                )
                out = hit
                break
    if out is None and _ACTION_START.match(t) and (not _NO_ACTION.search(t)):
        out = _cap(t)
    if out is None:
        if raw and raw not in _seen:
            _seen.add(raw)
            logger.info("[canon-miss] %r left unchanged", raw)
        return (raw, set())
    if raw not in _seen:
        _seen.add(raw)
        if out != raw:
            logger.info("[canon] %r -> %r", raw, out)
    return (out, set(skills_of.get(out, ())))
