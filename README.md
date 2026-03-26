# cadresearch

`cadresearch` is an `autoresearch`-style scaffold for the first stage of mesh-to-CAD:

- input: point clouds with normals sampled from triangle meshes
- output: per-point primitive labels plus primitive proxy parameters
- metric: a fixed scalar score on a locked validation set
- contract: the agent only edits `train.py`

This repo does **not** solve full Mesh -> STEP yet. It focuses on the tractable first loop: surface understanding for CAD reconstruction.

## Current scope

The initial task is:

- classify each point as `plane`, `cylinder`, `cone`, `sphere`, or `other`
- regress a compact primitive parameter vector for the points that belong to analytic surfaces

That makes the research loop fast, measurable, and hard to game.

## Project contract

The repo mirrors `autoresearch`:

- `build_dataset.py` builds fixed `.npz` shards once
- `prepare.py` is read-only infrastructure: constants, dataloaders, evaluation
- `train.py` is the only file the agent should modify
- `program.md` is the human-written research brief

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Build a synthetic starter dataset
uv run build_dataset.py synthetic

# 3. Verify the dataset and loader contract
uv run prepare.py

# 4. Run the baseline experiment
uv run train.py
```

## Autonomous loop

`research.py` is the first `autoresearch`-style runner for this repo. It:

- seeds itself from the current best run
- asks a coding agent to edit `train.py`
- optionally syncs `train.py` to another machine
- runs the fixed benchmark
- keeps the change only if `val_score` improves
- records all attempts under `artifacts/autoresearch/<run-name>/`

Seed a run from an existing best log:

```bash
python3 research.py seed \
  --run-name abc512 \
  --baseline-log artifacts/abc_improved_512.log
```

Then run one autonomous iteration locally with Codex:

```bash
python3 research.py loop \
  --run-name abc512 \
  --iterations 1 \
  --train-command 'CADRESEARCH_CACHE_DIR=artifacts/abc_cache_512 python3 train.py'
```

Or run edits locally and training on `bkawk.local`:

```bash
python3 research.py loop \
  --run-name abc512 \
  --iterations 4 \
  --pre-train-command 'scp train.py bkawk.local:/data/projects/mesh-para/cadresearch/train.py' \
  --train-command "ssh bkawk.local 'cd /data/projects/mesh-para/cadresearch && CADRESEARCH_CACHE_DIR=/data/projects/mesh-para/cadresearch/artifacts/abc_cache_512 PYTHONUNBUFFERED=1 python3 train.py'"
```

Check the current best at any time:

```bash
python3 research.py status --run-name abc512
```

The synthetic mode is only a bootstrap path so the loop is runnable immediately. For a real reverse-engineering dataset, preprocess your data into per-object `.npz` files and use the `pack` mode in `build_dataset.py`.

## Working With ABC

There is also a helper script for extracting a small matched raw subset from ABC archives:

```bash
python3 extract_abc_subset.py \
  --stl-archive /path/to/abc_0000_stl2_v00.7z \
  --step-archive /path/to/abc_0000_step_v00.7z \
  --feat-archive /path/to/abc_0000_feat_v00.7z \
  --output-dir /path/to/abc_subset \
  --limit 16
```

This creates:

- `manifest.json`
- `raw/<sample_id>/mesh.stl`
- `raw/<sample_id>/model.step`
- `raw/<sample_id>/features.yml`

That raw subset is the bridge from downloaded ABC archives to a future packer that emits `.npz` training shards.

To turn an extracted raw subset into per-object `.npz` files:

```bash
python3 preprocess_abc_raw.py \
  --input-dir /path/to/abc_subset \
  --output-dir /path/to/abc_preprocessed \
  --num-points 2048 \
  --val-count 1
```

Then pack those `.npz` files into train/val shards with:

```bash
uv run build_dataset.py pack --input-dir /path/to/abc_preprocessed
```

## Dataset format

Each packed sample represents one object and must contain:

- `points`: `float32[num_points, 3]`
- `normals`: `float32[num_points, 3]`
- `labels`: `int64[num_points]`
- `params`: `float32[num_points, 8]`
- `param_mask`: `float32[num_points, 8]`

The fixed parameter vector is:

1. position `x, y, z`
2. axis or normal `x, y, z`
3. value `0`
4. value `1`

Its semantics are class-specific:

- plane: center, normal, half-width, half-height
- cylinder: centerline midpoint, axis, radius, half-height
- cone: apex, axis, half-angle, height
- sphere: center, unused axis slots, radius, angular span
- other: ignored by `param_mask`

This is intentionally a proxy target, not a final BRep.

## Metric

`prepare.py` computes a fixed validation score:

```text
val_score = 0.7 * macro_iou + 0.3 * (1 / (1 + param_rmse_norm))
```

Higher is better.

## Roadmap

The scaffold is designed to grow in stages:

1. surface segmentation + primitive proxy regression
2. primitive instance grouping and fitting
3. topology assembly
4. BRep / STEP export

The autonomous loop should start at stage 1 and earn its way into later stages.
