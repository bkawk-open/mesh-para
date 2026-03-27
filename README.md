# mesh-para

`mesh-para` is an `autoresearch`-inspired project for the first stage of mesh-to-CAD.

The working idea is simple:

- input: triangle mesh -> sampled point cloud with normals
- output: per-point analytic surface labels, primitive proxy parameters, and boundary supervision
- goal: improve geometric understanding enough that later primitive fitting and CAD reconstruction become practical

This repo does not solve full Mesh -> STEP today. It focuses on the stage-1 problem that has a fast, fixed benchmark and can be searched overnight by an agent.

The repo pattern comes from the cloned [autoresearch reference repo](/Volumes/bkawk/projects/mesh-para/reference). We also borrowed useful workflow ideas from [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch), [`obra/superpowers`](https://github.com/obra/superpowers), and [`garrytan/gstack`](https://github.com/garrytan/gstack), while keeping this project much narrower than any of them. @matthew_berman helped put `autoresearch` on our radar in the first place.

## Why this exists

Mesh -> CAD is not a normal format conversion. The mesh has already thrown away design intent, and recovering CAD means inferring:

- surfaces
- boundaries
- primitive parameters
- eventually instances, topology, and valid BRep structure

This repo applies the `autoresearch` loop idea to the first learnable part of that pipeline.

## Current status

The project is past the synthetic scaffold stage:

- real ABC-backed datasets are working
- a guarded autonomous research loop is running
- the best gains so far came from better supervision, not just bigger models

Important score milestones:

- weak PointNet-style baseline: `0.137444`
- first strong local model: `0.329439`
- boundary-supervision baseline: `0.332057`
- current best promoted result: `0.362058`

The current benchmark is intentionally narrow:

```text
val_score = 0.7 * macro_iou + 0.3 * (1 / (1 + param_rmse_norm))
```

Higher is better.

## Shared lab

This repo is meant to become a small open research lab, not just a code dump.

The important thing to preserve is not only a winning patch, but the shared memory around it:

- what worked
- what almost worked
- what failed cleanly
- what hardware it ran on

That matters here because the project is strongly shaped by fixed wall-clock budget and hardware throughput. A result on one GPU may mean something different on another.

## Repo shape

The repo keeps the same core contract as the reference `autoresearch` pattern:

- [train.py](/Volumes/bkawk/projects/mesh-para/cadresearch/train.py): main editable research file
- [prepare.py](/Volumes/bkawk/projects/mesh-para/cadresearch/prepare.py): fixed loading and evaluation path
- [build_dataset.py](/Volumes/bkawk/projects/mesh-para/cadresearch/build_dataset.py): fixed shard-building path
- [program.md](/Volumes/bkawk/projects/mesh-para/cadresearch/program.md): human-written research brief
- [research.py](/Volumes/bkawk/projects/mesh-para/cadresearch/research.py): worker loop
- [manager.py](/Volumes/bkawk/projects/mesh-para/cadresearch/manager.py): lab-manager layer above the worker

## Start here

If you want the big picture:

- [Project Overview](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/overview.md)
- [Findings](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/findings.md)
- [Next Phase](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/next_phase.md)
- [Shared Lab Memory](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/shared_lab_memory.md)

If you want to run or operate the system:

- [Workflow](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/workflow.md)
- [Contributing](/Volumes/bkawk/projects/mesh-para/cadresearch/CONTRIBUTING.md)

If you want standardized, commit-friendly summaries of wins and misses:

- [Lab Notebook](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/lab_notebook)

## Quick start

```bash
cd /Volumes/bkawk/projects/mesh-para/cadresearch
uv sync
uv run build_dataset.py synthetic
uv run prepare.py
uv run train.py
```

For real data and autonomy details, use [Workflow](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/workflow.md).

## Autonomy

The repo has two layers:

- [research.py](/Volumes/bkawk/projects/mesh-para/cadresearch/research.py): one guarded keep-or-revert worker loop
- [manager.py](/Volumes/bkawk/projects/mesh-para/cadresearch/manager.py): strategy selection, promotion, audits, and relaunching

The cleanest status entrypoints are:

```bash
python3 /Volumes/bkawk/projects/mesh-para/cadresearch/manager.py \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  status --source-run boundary512_refocused
```

```bash
/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh status
```

For full operating details, use [Workflow](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/workflow.md).

To publish a run into the shared-lab format:

```bash
python3 /Volumes/bkawk/projects/mesh-para/cadresearch/manager.py \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  publish --run-name <run-name>
```

## Hardware

Faster GPUs matter a lot here because the benchmark is fixed by wall-clock time, not by step count.

That means stronger hardware can change which ideas are competitive:

- more optimization steps inside the same 5-minute budget
- more room for richer local-geometry models
- more overnight experiment throughput

The repo now includes a hardware-aware prep path:

```bash
python3 /Volumes/bkawk/projects/mesh-para/cadresearch/manager.py \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  hardware
```

That caches a hardware profile and lets the manager switch to [strategies_expanded](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies_expanded) automatically when the machine qualifies.

## Open source goal

One of the most exciting outcomes would be other people running this repo on bigger machines and contributing back:

- better `train.py` diffs
- promising near misses
- clean negative results
- hardware-specific wins

If that happens, the project starts to behave less like one local experiment loop and more like a shared research program.
