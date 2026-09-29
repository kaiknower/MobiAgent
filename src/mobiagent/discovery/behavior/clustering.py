import json
import re
from collections import defaultdict
from collections.abc import Iterable

from mobiagent.discovery.behavior.azure_client import build_chat_completion_request
from mobiagent.discovery.behavior.azure_client import execute_chat_completion
from mobiagent.discovery.behavior.azure_client import execute_chat_completion_with_provider
from mobiagent.discovery.behavior.azure_client import extract_first_message_text


class MissingClusterItemIdsError(ValueError):
    def __init__(self, missing_item_ids: list[str]) -> None:
        super().__init__(f"Named cluster response missing item_ids: {missing_item_ids}")
        self.missing_item_ids = missing_item_ids


def normalize_cluster_key(value: str) -> str:
    return value.strip().lower()


def cluster_identical_normalized_strings(values: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in values:
        grouped[normalize_cluster_key(value)].append(value)
    return dict(grouped)


def summarize_value_frequencies(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def build_description_clusters(predictions: list[dict]) -> list[dict]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for prediction in predictions:
        for skill in prediction.get("skill_timeline", []):
            description = skill.get("skill_description")
            if isinstance(description, str):
                grouped[normalize_cluster_key(description)].append(description)

    return [
        {
            "cluster_key": cluster_key,
            "count": len(descriptions),
            "descriptions": descriptions,
        }
        for cluster_key, descriptions in sorted(grouped.items())
    ]


def _activity_type_lookup(prediction: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for activity in prediction.get("activity_timeline", []):
        if not isinstance(activity, dict):
            continue
        activity_id = activity.get("activity_id")
        activity_type = activity.get("activity_type")
        if isinstance(activity_id, str) and isinstance(activity_type, str):
            lookup[activity_id] = activity_type
    return lookup


_NAVIGATION_LEADING_PATTERNS = (
    "move to",
    "move along",
    "move back",
    "move through",
    "move across",
    "move towards",
    "move toward",
    "move into",
    "move out",
    "navigate",
    "walk",
    "travel",
    "carry",
    "go to",
    "head to",
    "head toward",
    "head towards",
    "head back",
    "approach ",
    "return to",
    "return back",
)


def _fallback_activity_type(description: str) -> str:
    cleaned = description.strip().lower()
    for pattern in _NAVIGATION_LEADING_PATTERNS:
        if cleaned.startswith(pattern):
            return "navigation"
    return "manipulation"


def _description_implies_navigation(description: str) -> bool:
    cleaned = description.strip().lower()
    for pattern in _NAVIGATION_LEADING_PATTERNS:
        if cleaned.startswith(pattern):
            return True
    return False


def build_skill_description_items(predictions: list[dict]) -> list[dict]:
    items: list[dict] = []
    for prediction_index, prediction in enumerate(predictions):
        task_id = str(prediction.get("task_id", f"task-{prediction_index:04d}"))
        episode_id = str(prediction.get("episode_id", f"episode-{prediction_index:06d}"))
        activity_types = _activity_type_lookup(prediction)
        for segment_index, segment in enumerate(prediction.get("skill_timeline", []), start=1):
            if not isinstance(segment, dict):
                continue
            description = segment.get("skill_description")
            if not isinstance(description, str) or not description.strip():
                continue
            segment_id = str(segment.get("segment_id", f"segment-{segment_index:03d}"))
            parent_activity_id = segment.get("parent_activity_id")
            activity_type = (
                activity_types.get(parent_activity_id)
                if isinstance(parent_activity_id, str)
                else None
            )
            if activity_type not in {"navigation", "manipulation"}:
                activity_type = _fallback_activity_type(description)
            # Description-based override: if the description clearly starts with a
            # navigation verb, trust the visible description over the parent activity's
            # label. This catches inference-level mistakes where a whole activity was
            # wrongly typed as manipulation despite its skills being travel.
            if activity_type != "navigation" and _description_implies_navigation(description):
                activity_type = "navigation"
            item_id = f"item-{len(items) + 1:06d}"
            items.append(
                {
                    "item_id": item_id,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "segment_id": segment_id,
                    "activity_type": activity_type,
                    "description": description,
                }
            )
    return items


def build_named_description_clusters(
    clusters: list[dict],
    name_lookup: dict[str, str],
) -> list[dict]:
    return [
        {
            **cluster,
            "canonical_name": name_lookup.get(cluster["cluster_key"], cluster["cluster_key"]),
        }
        for cluster in clusters
    ]


def _extract_first_json_value(text: str) -> str:
    start_positions = [index for index in (text.find("["), text.find("{")) if index != -1]
    if not start_positions:
        raise ValueError("No JSON value found")

    start = min(start_positions)
    opening = text[start]
    closing = "]" if opening == "[" else "}"
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("No complete JSON value found")


def build_cluster_naming_request(
    items: list[dict],
    model: str,
    max_completion_tokens: int,
) -> dict:
    prompt = "\n".join(
        [
            "## Overview",
            "Semantically cluster inference-produced skill-segment action descriptions, then name each cluster with a reusable canonical action-intent name.",
            "",
            "## Inputs",
            "- You receive a flat list of skill description items, not precomputed semantic clusters.",
            "- Each item corresponds to one observed skill segment.",
            "- Each item has `item_id`, `task_id`, `episode_id`, `segment_id`, `activity_type`, and `description`.",
            "- `item_id` is an opaque stable identifier; copy it exactly and do not derive new IDs from task_id, episode_id, or segment_id.",
            "- `activity_type` is copied from inference's coarse activity_timeline and is either navigation or manipulation.",
            "- `description` is a skill-segment action description produced by inference, not a canonical name.",
            "- Descriptions intentionally include concrete context needed to describe the observed segment.",
            "- Descriptions may include object, destination, receptacle, support surface, spatial relation, room, furniture, or scene details.",
            "- Use contextual words only to understand the action intent.",
            "- Do not use contextual words as canonical split criteria.",
            "",
            "## Required Two-Step Procedure",
            "Step 1: Semantic clustering",
            "- Group items by action intent.",
            "- Use activity_type as a strong grouping boundary.",
            "- Do not merge navigation items with manipulation items.",
            "- For manipulation, group by state-changing intent, not by object, room, container, support, or surface wording.",
            "- Use object/location context only to understand the action, not to split groups.",
            "",
            "Step 2: Canonical naming",
            "- Assign one canonical_name to each semantic group.",
            "- canonical_name is an abstract action-intent label.",
            "- canonical_name should be short and reusable.",
            "- canonical_name should not include task-specific objects, rooms, scene names, or support surfaces unless needed to distinguish action type.",
            "",
            "## Hard Rules",
            "- Return valid JSON only.",
            "- Return a JSON array only.",
            "- Do not add markdown fences or explanations.",
            "- Every output item must contain: `canonical_name`, `activity_type`, `item_ids`, and `descriptions`.",
            "- Every input item_id must appear exactly once across the full output.",
            "- Do not omit any item because it is ambiguous, mixed, duplicated, low quality, or difficult to name.",
            "- If an item is ambiguous or mixed, still assign it to the best available action-intent group exactly once.",
            "- Preserve input descriptions exactly.",
            "- Do not paraphrase, shorten, normalize, or edit descriptions.",
            "- Do not invent item_ids or descriptions.",
            "- Do not invent object names or environment details.",
            "",
            "## Naming Policy",
            "- For items whose activity_type is navigation and whose primary intent is travel or relocation, canonical_name must be exactly `move to`.",
            "- Movement while holding an object remains `move to` when the item is navigation.",
            "- Do not apply the navigation rule to local movement inside manipulation items; those should be grouped under the supported state-changing manipulation intent.",
            "- Group manipulation by state-changing intent.",
            "- Preparation/contact phases should not become canonical names by themselves when they only support another state-changing intent.",
            "- Completion/recovery phases should not become canonical names by themselves when they only finish another state-changing intent.",
            "- Map preparation/contact/completion/recovery descriptions to the canonical action intent they support when possible.",
            "- If an item contains mixed intents, group by the primary state-changing intent or navigation intent.",
            "- Use short verb phrases.",
            "- Prefer action-level names over scene-specific names.",
            "- Prefer GENERAL action categories over specific physical variants. Do not create separate canonical_names for fine-grained ways of placing (stack / pile / lay-on-top / drop-on / rest-on) or acquiring (retrieve / fetch / extract) or actuating (pull-open / swing-shut) — collapse each variant into its general category. A skill label should describe WHAT action type is performed, not how finely the robot executed it.",
            "- canonical_name must be lowercase.",
            "- Do not output two groups with the same canonical_name. If two groups would receive the same canonical_name, merge them into a single group.",
            "- Do not produce canonical_names that are synonyms of each other (for example `place at` and `place on`, or `set down` and `place on`). Pick ONE canonical surface form per action intent and group all synonyms under it.",
            "- Do not produce canonical_names for idleness, completion, preparation, recovery, adjustment, waiting, finalization, standing by, observation, alignment, arrangement, or other non-state-changing filler phases. Always merge each such item into the nearest state-changing manipulation cluster — pick whichever existing canonical_name best captures the action that the filler supports (preparation of an acquire → `pick up from`; preparation or completion of a placement → the matching `place on` / `place in`; an alignment before an `open` → `open`; etc.).",
            "- Do NOT create a catch-all `other` cluster. If you are tempted to, you have not chosen the nearest state-changing intent; go back and assign the item to one of the real action clusters.",
            "- Synonym collapse rules (apply strictly):",
            "- `place at`, `place onto`, `set down`, `set on`, `put on`, `put down`, `drop onto`, `lay on`, `stack on`, `stack onto`, `stack on top of`, `pile on`, `rest on` → `place on` (any action that ends with an object resting on a surface, another object, or a stack is `place on`; do not create a separate `stack` category)",
            "- `place into`, `place inside`, `put in`, `put inside`, `insert`, `insert into`, `insert X into Y`, `drop into`, `load into`, `slide into`, `tilt the plate to slide into`, `pour into`, `tip the plate to slide into` → `place in` (any action that ends with an object landing inside a container/receptacle — by direct release, by tilting a held surface so the object slides off into the container, or by pouring — is `place in`; never create a separate `insert`, `slide`, `tip`, or `tilt` cluster).",
            "- `lift`, `grasp from`, `grasp`, `grab`, `take from`, `retrieve`, `acquire` → `pick up from`",
            "- `pull open`, `swing open`, `push open` → `open`",
            "- `push close`, `swing shut`, `pull shut` → `close`",
            "- `move away from`, `back away from`, `leave`, `retreat from`, `depart from`, `move back from`, `walk away from`, `step away from`, `move aside` → `move to` (any base travel is navigation under `move to`; never create a `move away` / `retreat` / `leave` cluster — force the item into `move to` regardless of whether the description is phrased as departure). Set activity_type to `navigation` for these items.",
            "- Any fine-grained variant with a destination surface maps to `place on`; any fine-grained variant with a destination container/receptacle maps to `place in`. Never keep a one-member cluster whose description is a variant of placing — always fold it into the matching `place on` / `place in`.",
            "- Destination-aware routing for release-style verbs:",
            "- If the description reads `release/drop/let go of ... onto ...` or `... on the [surface]`, canonical_name is `place on`.",
            "- If the description reads `release/drop/let go of ... into ...` or `... in the [container]`, canonical_name is `place in`.",
            "- Only use `release` when there is no destination surface or container in the description (e.g., pure handoff or letting-go in free space). Prefer mapping to `place on` or `place in` whenever a destination is mentioned.",
            "- Style examples only, not a closed vocabulary:",
            "- pick up from",
            "- place on",
            "- place in",
            "- open",
            "- close",
            "- insert",
            "- remove",
            "- wipe",
            "- toggle",
            "- release",
            "- push",
            "- pull",
            "- If none of the style examples fit, choose another short action-oriented canonical name.",
            "",
            "## Response Contract",
            "Use this output shape:",
            "Field values in this JSON shape are type placeholders only; do not copy placeholder values.",
            (
                '[{"canonical_name":"short reusable action-intent label",'
                '"activity_type":"navigation | manipulation",'
                '"item_ids":["input item_id"],'
                '"descriptions":["original input description"]}]'
            ),
            "",
            "Input skill items:",
            json.dumps(items, sort_keys=True),
        ]
    )
    return build_chat_completion_request(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You semantically cluster skill-segment action descriptions and assign reusable canonical action-intent names. "
                    "Return a JSON array only. "
                    "Do not add markdown fences or explanations."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_completion_tokens=max_completion_tokens,
    )


def build_cluster_naming_retry_request(
    *,
    items: list[dict],
    previous_groups: list[dict],
    missing_item_ids: list[str],
    model: str,
    max_completion_tokens: int,
) -> dict:
    prompt = "\n".join(
        [
            "Your previous semantic clustering response was invalid.",
            "Previous response was invalid because it omitted these item_ids:",
            json.dumps(missing_item_ids, sort_keys=True),
            "",
            "Return a corrected complete JSON array.",
            "Every input item_id must appear exactly once across the full output.",
            "Do not omit ambiguous, mixed, duplicated, low-quality, or difficult items.",
            "If an item is ambiguous or mixed, assign it to the best available action-intent group exactly once.",
            "Preserve input descriptions exactly.",
            "You may reuse or revise the previous groups, but the corrected response must satisfy the contract.",
            "",
            "Previous groups:",
            json.dumps(previous_groups, sort_keys=True),
            "",
            "Input skill items:",
            json.dumps(items, sort_keys=True),
        ]
    )
    return build_chat_completion_request(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You correct semantic skill clustering JSON. "
                    "Return a JSON array only. "
                    "Do not add markdown fences or explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_completion_tokens,
    )


def parse_named_clusters_response_text(text: str) -> list[dict]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        preview = text.strip().replace("\n", "\\n")
        if len(preview) > 500:
            preview = preview[:500] + "..."
        payload = json.loads(_extract_first_json_value(text))

    if not isinstance(payload, list):
        raise ValueError("Named cluster response must be a list")
    return payload


def _validate_named_clusters(named_clusters: list[dict], items: list[dict]) -> list[dict]:
    expected_item_ids = {item["item_id"] for item in items}
    item_activity_type_by_id = {item["item_id"]: item.get("activity_type") for item in items}
    seen_item_ids: set[str] = set()
    validated: list[dict] = []
    for item in named_clusters:
        if not isinstance(item, dict):
            raise ValueError("Each named cluster item must be an object")
        canonical_name = item.get("canonical_name")
        activity_type = item.get("activity_type")
        item_ids = item.get("item_ids")
        descriptions = item.get("descriptions")
        if not isinstance(canonical_name, str):
            raise ValueError("canonical_name must be a string")
        if not isinstance(activity_type, str):
            raise ValueError("activity_type must be a string")
        if not isinstance(item_ids, list) or not all(isinstance(item_id, str) for item_id in item_ids):
            raise ValueError("item_ids must be a list of strings")
        if not isinstance(descriptions, list) or not all(isinstance(desc, str) for desc in descriptions):
            raise ValueError("descriptions must be a list of strings")
        for item_id in item_ids:
            if item_id not in expected_item_ids:
                raise ValueError(f"Unknown item_id in named cluster response: {item_id}")
            if item_id in seen_item_ids:
                raise ValueError(f"Duplicate item_id in named cluster response: {item_id}")
            seen_item_ids.add(item_id)
        # Override cluster-level activity_type from the member items (majority vote),
        # so clusters can't mislabel navigation items as manipulation or vice versa.
        member_types = [item_activity_type_by_id.get(item_id) for item_id in item_ids]
        member_types = [t for t in member_types if t in {"navigation", "manipulation"}]
        if member_types:
            nav_count = sum(1 for t in member_types if t == "navigation")
            manip_count = len(member_types) - nav_count
            majority = "navigation" if nav_count >= manip_count else "manipulation"
            activity_type = majority
        validated.append(
            {
                "canonical_name": canonical_name,
                "activity_type": activity_type,
                "item_ids": item_ids,
                "descriptions": descriptions,
            }
        )
    missing_item_ids = expected_item_ids - seen_item_ids
    if missing_item_ids:
        raise MissingClusterItemIdsError(sorted(missing_item_ids))
    validated = _drop_other_cluster(validated)
    return validated


def _drop_other_cluster(validated: list[dict]) -> list[dict]:
    """Drop any `other` catch-all cluster from the cluster output.

    The prompt forbids creating an `other` cluster — it is reserved for
    preparation/observation/alignment/filler items that are not real state
    changes. If the LLM still emits one, we remove it from the cluster output
    so downstream consumers only see real action clusters. Items from the
    dropped cluster remain in `semantic_skill_items.json` and per-task
    timelines; they simply do not contribute to canonical cluster groups.
    """
    return [c for c in validated if c["canonical_name"].strip().lower() != "other"]


_CANONICAL_ALIAS = {
    "place at": "place on",
    "place onto": "place on",
    "set down": "place on",
    "set on": "place on",
    "set onto": "place on",
    "put on": "place on",
    "put down": "place on",
    "drop onto": "place on",
    "lay on": "place on",
    "lay down": "place on",
    "place into": "place in",
    "place inside": "place in",
    "put in": "place in",
    "put inside": "place in",
    "insert into": "place in",
    "lift": "pick up from",
    "grasp from": "pick up from",
    "grasp": "pick up from",
    "grab": "pick up from",
    "take from": "pick up from",
    "pick up": "pick up from",
    "acquire": "pick up from",
    "retrieve": "pick up from",
    "fetch": "pick up from",
    "pull open": "open",
    "swing open": "open",
    "push open": "open",
    "push close": "close",
    "push closed": "close",
    "swing shut": "close",
    "pull shut": "close",
    "close shut": "close",
    "release": "place in",  # destinationless release of a carried task object defaults to place in
}

_CANONICAL_DROP = {
    "finalize",
    "finish",
    "complete",
    "completion",
    "idle",
    "stand by",
    "standby",
    "wait",
    "stop",
    "adjust",
    "adjust grip",
    "reposition",
    "prepare",
    "preparation",
    "recover",
    "recovery",
    "observe",
    "observation",
    "check",
    "settle",
    "stabilize",
    "stabilise",
}


_RELEASE_VERBS = ("release", "drop", "let go")
_ONTO_HINT = re.compile(r"\b(onto|on the|on top of|lay on|set on)\b", re.IGNORECASE)
_INTO_HINT = re.compile(r"\b(into|in the|inside the|insert into)\b", re.IGNORECASE)


def _normalize_canonical_name(raw: str) -> str:
    key = (raw or "").strip().lower()
    return _CANONICAL_ALIAS.get(key, key)


def _reroute_release_description(description: str) -> str | None:
    """If a release-style description has a visible destination, return its target canonical_name."""
    if not isinstance(description, str):
        return None
    lower = description.lower()
    if not any(verb in lower for verb in _RELEASE_VERBS):
        return None
    if _INTO_HINT.search(lower):
        return "place in"
    if _ONTO_HINT.search(lower):
        return "place on"
    return None


def normalize_canonical_clusters(groups: list[dict]) -> list[dict]:
    """Collapse synonym duplicates and drop spurious canonical_names after LLM naming."""

    merged: dict[tuple[str, str], dict] = {}

    def _fallback_for(activity_type: str) -> dict:
        key = ("other", activity_type)
        if key not in merged:
            merged[key] = {
                "canonical_name": "other",
                "activity_type": activity_type,
                "item_ids": [],
                "descriptions": [],
            }
        return merged[key]

    def _bucket(canonical: str, activity_type: str) -> dict:
        key = (canonical, activity_type)
        if key not in merged:
            merged[key] = {
                "canonical_name": canonical,
                "activity_type": activity_type,
                "item_ids": [],
                "descriptions": [],
            }
        return merged[key]

    for group in groups:
        if not isinstance(group, dict):
            continue
        raw_name = group.get("canonical_name", "")
        activity_type = group.get("activity_type", "manipulation")
        canonical = _normalize_canonical_name(raw_name)
        item_ids = list(group.get("item_ids", []))
        descriptions = list(group.get("descriptions", []))

        pairs = list(zip(item_ids, descriptions))
        # Pad missing descriptions so we can still place the item in a bucket.
        if len(item_ids) > len(descriptions):
            pairs = list(zip(item_ids, descriptions + [""] * (len(item_ids) - len(descriptions))))

        for item_id, description in pairs:
            item_canonical = canonical
            if item_canonical == "release" or item_canonical in _CANONICAL_DROP:
                rerouted = _reroute_release_description(description)
                if rerouted:
                    item_canonical = rerouted

            if item_canonical in _CANONICAL_DROP or item_canonical == "":
                target = _fallback_for(activity_type)
            else:
                target = _bucket(item_canonical, activity_type)
            target["item_ids"].append(item_id)
            target["descriptions"].append(description)

    return list(merged.values())


_VERB_FALLBACK_RULES: list[tuple[tuple[str, ...], str, str]] = [
    # (verb-prefixes, canonical_name, activity_type)
    (("move to", "move toward", "move away", "move back", "walk to", "walk toward", "back away", "leave", "retreat", "depart"), "move to", "navigation"),
    (("pick up", "grasp", "grab", "lift", "acquire", "take from", "retrieve", "fetch"), "pick up from", "manipulation"),
    (("place in", "place into", "put in", "put into", "insert", "drop into", "release into", "release in"), "place in", "manipulation"),
    (("place on", "place onto", "put on", "put down", "set down", "drop onto", "release onto", "release on"), "place on", "manipulation"),
    (("release", "let go", "drop", "hand off"), "place in", "manipulation"),
    (("open",), "open", "manipulation"),
    (("close", "shut"), "close", "manipulation"),
]


def _fallback_assign_missing_items(
    validated: list[dict],
    items: list[dict],
    missing_item_ids: list[str],
) -> list[dict]:
    """Force-assign items still unassigned after LLM retries by matching the
    leading verb in the description to a canonical cluster. Any item that
    still cannot be matched goes into an activity-type-aware fallback cluster."""
    items_by_id = {item["item_id"]: item for item in items}
    clusters_by_name: dict[str, dict] = {c["canonical_name"]: c for c in validated}

    def _ensure_cluster(name: str, activity_type: str) -> dict:
        if name not in clusters_by_name:
            cluster = {
                "canonical_name": name,
                "activity_type": activity_type,
                "item_ids": [],
                "descriptions": [],
            }
            clusters_by_name[name] = cluster
            validated.append(cluster)
        return clusters_by_name[name]

    for item_id in missing_item_ids:
        item = items_by_id.get(item_id)
        if item is None:
            continue
        description = (item.get("description") or "").strip().lower()
        activity_type = item.get("activity_type") or "manipulation"
        assigned = False
        for prefixes, canonical, default_type in _VERB_FALLBACK_RULES:
            if any(description.startswith(p) for p in prefixes):
                target = _ensure_cluster(canonical, default_type)
                target["item_ids"].append(item_id)
                target["descriptions"].append(item.get("description") or "")
                assigned = True
                break
        if not assigned:
            fallback_name = "move to" if activity_type == "navigation" else "pick up from"
            fallback_type = activity_type if activity_type in {"navigation", "manipulation"} else "manipulation"
            target = _ensure_cluster(fallback_name, fallback_type)
            target["item_ids"].append(item_id)
            target["descriptions"].append(item.get("description") or "")
    return validated


def run_cluster_naming(
    items: list[dict],
    model: str,
    max_completion_tokens: int,
    client: object | None = None,
) -> list[dict]:
    request = build_cluster_naming_request(
        items=items,
        model=model,
        max_completion_tokens=max_completion_tokens,
    )
    response = execute_chat_completion_with_provider(request, client=client, provider="azure")
    named_clusters = parse_named_clusters_response_text(extract_first_message_text(response))
    try:
        validated = _validate_named_clusters(named_clusters, items)
    except MissingClusterItemIdsError as missing_error:
        retry_request = build_cluster_naming_retry_request(
            items=items,
            previous_groups=named_clusters,
            missing_item_ids=missing_error.missing_item_ids,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
        retry_response = execute_chat_completion_with_provider(retry_request, client=client, provider="azure")
        retry_named_clusters = parse_named_clusters_response_text(extract_first_message_text(retry_response))
        try:
            validated = _validate_named_clusters(retry_named_clusters, items)
        except MissingClusterItemIdsError as retry_missing_error:
            # LLM still unable to assign — start from the original clusters
            # (which have most items), then force-assign stragglers via verb match.
            partial_validated = _validate_named_clusters_lenient(named_clusters, items)
            already_assigned = {
                item_id for cluster in partial_validated for item_id in cluster["item_ids"]
            }
            still_missing = [
                i for i in retry_missing_error.missing_item_ids if i not in already_assigned
            ]
            all_expected = {item["item_id"] for item in items}
            still_missing.extend(i for i in all_expected if i not in already_assigned and i not in still_missing)
            validated = _fallback_assign_missing_items(
                partial_validated, items, sorted(still_missing)
            )
    return normalize_canonical_clusters(validated)


def _validate_named_clusters_lenient(named_clusters: dict, items: list[dict]) -> list[dict]:
    """Like _validate_named_clusters but allows missing item_ids (returns what
    could be validated). Used only as a pre-step before fallback assignment."""
    expected_item_ids = {item["item_id"] for item in items}
    item_activity_type_by_id = {item["item_id"]: item.get("activity_type") for item in items}
    validated: list[dict] = []
    seen_item_ids: set[str] = set()
    groups = named_clusters.get("groups", []) if isinstance(named_clusters, dict) else []
    if not isinstance(groups, list):
        return validated
    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical_name = group.get("canonical_name")
        activity_type = group.get("activity_type")
        item_ids = group.get("item_ids")
        descriptions = group.get("descriptions")
        if not isinstance(canonical_name, str) or not isinstance(activity_type, str):
            continue
        if not isinstance(item_ids, list) or not isinstance(descriptions, list):
            continue
        filtered_item_ids = [
            i for i in item_ids
            if isinstance(i, str) and i in expected_item_ids and i not in seen_item_ids
        ]
        filtered_descriptions = [d for d in descriptions if isinstance(d, str)][: len(filtered_item_ids)]
        for item_id in filtered_item_ids:
            seen_item_ids.add(item_id)
        member_types = [item_activity_type_by_id.get(i) for i in filtered_item_ids]
        member_types = [t for t in member_types if t in {"navigation", "manipulation"}]
        if member_types:
            nav_count = sum(1 for t in member_types if t == "navigation")
            manip_count = len(member_types) - nav_count
            activity_type = "navigation" if nav_count >= manip_count else "manipulation"
        validated.append({
            "canonical_name": canonical_name,
            "activity_type": activity_type,
            "item_ids": filtered_item_ids,
            "descriptions": filtered_descriptions,
        })
    return _drop_other_cluster(validated)


def summarize_timeline_skill_descriptions(predictions: list[dict]) -> dict[str, int]:
    descriptions: list[str] = []
    for prediction in predictions:
        for skill in prediction.get("skill_timeline", []):
            description = skill.get("skill_description")
            if isinstance(description, str):
                descriptions.append(description)
    return summarize_value_frequencies(descriptions)


__all__ = [
    "build_description_clusters",
    "build_cluster_naming_request",
    "build_cluster_naming_retry_request",
    "build_named_description_clusters",
    "build_skill_description_items",
    "cluster_identical_normalized_strings",
    "normalize_cluster_key",
    "parse_named_clusters_response_text",
    "run_cluster_naming",
    "summarize_timeline_skill_descriptions",
    "summarize_value_frequencies",
]
