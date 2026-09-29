# MobiAgent workflow skills

For simulation execution, demonstration processing or policy training requests,
read the relevant skill before choosing commands. Resolve paths from the repository
root and use the user's selected environment and assets.

| Request | Skill |
| --- | --- |
| Execute a RoboCasa or BEHAVIOR task; serve a checkpoint | [MobiAgent Execution](skills/mobiagent-execution/SKILL.md) |
| Discover BEHAVIOR skills; build training segments and expert shards | [MobiAgent Discovery](skills/mobiagent-discovery/SKILL.md) |
| Train or resume RoboCasa or BEHAVIOR policies | [MobiAgent Training](skills/mobiagent-training/SKILL.md) |

The skill files describe agent workflows. Their commands call the implementation
in `src/mobiagent/`, `scripts/`, and `policy/openpi/`. For code edits, keep the
corresponding skill and documentation consistent when an entrypoint or data
contract changes.
