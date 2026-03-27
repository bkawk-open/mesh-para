# Contributing

`mesh-para` works best when contributions carry research context, not just code.

The goal is not only to merge a better [train.py](/Volumes/bkawk/projects/mesh-para/cadresearch/train.py). The goal is to preserve shared research memory so other people can learn from:

- wins
- near misses
- dead ends
- hardware differences

## High-value contributions

The most useful contributions right now are:

- `train.py` variants that beat the current benchmark
- better supervision ideas for stage 1
- stronger strategy packs for the manager
- reproducible benchmark and audit results on different hardware
- documentation that explains why something worked or failed

## If you found a better model

Please include:

1. the `train.py` diff
2. the benchmark result on the main 512-set run
3. the larger audit result if you ran one
4. the hardware used
5. a short note on what changed and why you think it helped

Useful numbers to report:

- `val_score`
- `macro_iou`
- `param_rmse_norm`
- `param_score`
- training wall time
- VRAM usage if available

## If you found a useful near miss

Near misses are still valuable.

Please include:

- the idea that almost worked
- how close it got
- what got better
- what regressed
- whether the failure looked like:
  - throughput loss
  - instability
  - overfitting
  - worse class balance

## If you found a dead end

Negative results are welcome when they are legible.

Please include:

- what you tried
- why you thought it might help
- how it failed
- whether you think it is fundamentally bad or just bad on current hardware or budget

## Hardware matters

This repo is fixed by wall-clock budget, so hardware changes the effective search space.

Please report:

- GPU model
- GPU count
- VRAM
- driver version if relevant
- whether you used the default [strategies](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies) pack or the larger [strategies_expanded](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies_expanded) pack

If you ran:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch hardware
```

include that summary in the PR description.

## Good PR shape

A strong PR usually has:

- one focused code change or one focused strategy-pack change
- benchmark evidence
- hardware context
- a short plain-English explanation

If possible, generate a standardized summary first:

```bash
python3 /Volumes/bkawk/projects/mesh-para/cadresearch/manager.py \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  publish --run-name <run-name>
```

That writes a normalized markdown and JSON summary under [docs/lab_notebook](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/lab_notebook).

The easiest PRs to review are the ones that make it obvious whether the contribution is:

- a new best result
- a promising branch
- or a documented dead end

## Shared memory

Before opening a PR, it helps to skim:

- [README.md](/Volumes/bkawk/projects/mesh-para/cadresearch/README.md)
- [docs/findings.md](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/findings.md)
- [docs/next_phase.md](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/next_phase.md)
- [docs/shared_lab_memory.md](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/shared_lab_memory.md)

The more we can encode hits and misses as shared memory, the more this repo behaves like a real distributed research effort instead of isolated local experiments.
