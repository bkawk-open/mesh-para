# Project Overview

## Problem

This project is about recovering CAD-relevant structure from meshes.

The long-term target is:

```text
triangle mesh -> analytic surfaces -> primitive instances -> topology -> CAD / STEP
```

The current repo only addresses the first step well enough to support an autonomous research loop.

## Origin

This repo was bootstrapped from the design pattern in the cloned [autoresearch reference repo](/Volumes/bkawk/projects/mesh-para/reference).

What we borrowed:

- one editable training file
- fixed wall-clock experiment budget
- fixed evaluator
- human-written research brief
- autonomous keep-or-revert loop

What changed:

- language-model pretraining became mesh surface understanding
- the metric changed from language validation loss to geometric proxy quality
- the data pipeline changed from text shards to ABC-derived CAD supervision

## Current task

Each training sample is one object represented as:

- point cloud
- normals
- per-point surface class
- per-point proxy primitive parameters
- per-point boundary target

Current classes:

- `plane`
- `cylinder`
- `cone`
- `sphere`
- `other`

## Why this decomposition

Direct mesh -> STEP generation is too brittle for the current loop:

- evaluation would be slow
- invalid outputs would be common
- topology and CAD validity would dominate the signal

Instead, the repo uses a stage-1 metric that is fast and stable enough to optimize repeatedly.

## Benchmark shape

The loop is intentionally narrow:

- one editable training file
- one fixed evaluator
- fixed wall-clock budget
- single scalar score

This makes overnight autonomous experimentation possible.

## Datasets

The project currently uses:

- synthetic bootstrap data for smoke tests
- real ABC-derived mesh/STEP/features subsets for meaningful experiments

Important real subsets created so far:

- 512-sample set: main fast benchmark
- 1024-sample set: tested, not promoted
- 2048-sample set: secondary audit set

## What success looks like

Short term success:

- better `macro_iou`
- better `val_score`
- improvements that survive repeat runs

Longer term success:

- stage-1 model quality strong enough to support primitive fitting and topology work
- eventually a downstream metric that reflects real reconstruction quality, not just proxy labels
