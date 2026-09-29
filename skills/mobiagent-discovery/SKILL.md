---
name: mobiagent-discovery
description: Discover and describe manipulation skills from BEHAVIOR demonstration videos and state trajectories. Use for temporal skill annotation and semantic grouping before preparing per-expert training data.
---

# MobiAgent Skill Discovery

Run commands from the repository root, two levels above this file.
Read [data preparation](../../docs/data.md) for dependencies.

## Inputs

- BEHAVIOR dataset root containing `videos/`, `data/`, and `meta/`.
- Task IDs to process and an output root.
- Configured Azure OpenAI or Gemini provider credentials.

## Workflow

1. Install the discovery dependencies (`pip install -e '.[discovery]'`) in the
   working environment and ensure `ffmpeg` is available. Inspect the selected
   tasks' video, state and episode metadata before requesting model inference.
2. Run the offline pipeline for the requested task IDs:

   ```bash
   mobiagent-discover --platform behavior --dataset-root /path/to/behavior \
     --tasks task-0001 task-0003 --output-root outputs/discovery/behavior
   ```

3. Read the returned `inference_status`, `selected_demo_count`, `run_dir`, and
   artifact paths. `skipped_no_api_key`, `completed_no_selected_demos`, and `failed`
   do not establish that skills were successfully inferred. Investigate the
   reported condition before retrying; keep successful artifacts from partial runs.
4. Review temporal boundaries against the video's time scale and source trajectory.
   Keep skill descriptions grounded in observed actions. Inspect the pipeline's
   clustering status separately from temporal inference status.

## Outputs

Use the paths returned by the pipeline; do not assume a `latest/` directory exists.
A run includes:

- `manifests/selected_demos.json`: selected episodes.
- `predictions/demo_skill_predictions.jsonl`: temporal skill predictions.
- `review/`: timeline and readable annotations.
- `final/demo_skills.jsonl`: skill records.
- `clusters/`: frequency and semantic grouping artifacts.

When training data is requested next, pass the predictions, original dataset root
and task-instruction mapping to [MobiAgent Data Preparation](../mobiagent-data/SKILL.md).
The pipeline does not create the task-instruction mapping expected by the segment
builder; supply that mapping separately.

## Implementation

[Discovery pipeline](../../src/mobiagent/discovery/behavior/pipeline.py) and
[CLI](../../src/mobiagent/discovery/__main__.py).
