# Findings

## Main lessons so far

### 1. Simple fast models beat richer ones under a fixed 5-minute budget

Many heavier local-geometry ideas lost because they reduced optimization throughput too much.

Examples that did not beat the main fast baseline:

- geometry-aware edge encoder
- deeper two-block local message passing
- raw geometric edge features without supervision changes
- cosine schedule with warmup

### 2. Better supervision mattered more than fancier architecture

The first meaningful breakthrough after the initial local model came from boundary supervision.

Boundary-supervision baseline:

- `val_score 0.332057`
- `macro_iou 0.295052`

That beat the earlier non-boundary best:

- `val_score 0.329439`
- `macro_iou 0.286779`

### 3. The autonomous loop became useful only after the search space narrowed

Early on, the loop would have been close to blind search.

After:

- identifying the right fast model family
- proving heavier models often lose
- adding boundary supervision

the loop started finding real keepers.

### 4. More data does not automatically help under the fixed budget

Measured on the same 5-minute budget:

- `512` samples gave the best overall `val_score`
- `1024` samples was worse
- `2048` samples improved parameter fitting but hurt macro-IoU too much

Interpretation:

- larger data helps some aspects of generalization
- but the fixed wall-clock budget makes optimization/sample-efficiency the bottleneck

### 5. Longer budgets help the fast baseline, but did not rescue richer models

A 15-minute run improved the fast baseline, but the heavier geo-edge candidate still did not overtake it.

Interpretation:

- some regressions were budget-shaped
- but not all richer architectures were secretly better

## Current best known results

### Before boundary supervision

Best non-boundary local model:

- `val_score 0.329439`
- `macro_iou 0.286779`
- `param_rmse_norm 1.331115`

Source:

- [abc_improved_512.log](/Volumes/bkawk/projects/mesh-para/artifacts/abc_improved_512.log)

### Boundary-supervision baseline

- `val_score 0.332057`
- `macro_iou 0.295052`
- `param_rmse_norm 1.390053`

Source:

- [abc_boundary_512.log](/Volumes/bkawk/projects/mesh-para/artifacts/abc_boundary_512.log)

### Overnight loop best so far

- `val_score 0.359468`
- `macro_iou 0.330501`
- `param_rmse_norm 1.341591`

Source:

- [iter_0002_train.log](/Volumes/bkawk/projects/mesh-para/artifacts/autoresearch/boundary512/logs/iter_0002_train.log)

## What this means

The repo is now in a healthier place:

- supervision is better aligned with the task
- the model family is more stable
- the autonomous loop has a stronger baseline to search around

The next uncertainty is no longer “can the loop work at all?”
It is “how much further can boundary-aware stage-1 supervision take us before we need a new representation for stage 2?”

## What better hardware would mean

### A single RTX 5090

For this project, a 5090 would likely matter a lot.

The benchmark is fixed by wall-clock time, not by step count. That means faster hardware changes the effective search landscape:

- more optimization steps per experiment
- more headroom for local-neighborhood computation
- more realistic chances for richer models that currently lose on throughput

So a 5090 would not just speed up the current winner. It could change which architectures are viable under the same benchmark.

### Two RTX 5090s

Two cards would help most if treated as two experiment workers:

- one candidate per GPU
- parallel guarded experiments
- shared leaderboard and best-model promotion

That is much more attractive than adding multi-GPU distributed training complexity to a repo that is intentionally shaped around simplicity.
