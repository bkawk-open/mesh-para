# mesh-para

`mesh-para` is an `autoresearch`-inspired project for the first stage of mesh-to-CAD:

- input: triangle mesh -> sampled point cloud with normals
- output: per-point analytic surface labels, primitive proxy parameters, and boundary supervision
- goal: improve geometric understanding enough that later primitive fitting and CAD reconstruction become practical

This repo does not solve full Mesh -> STEP today. It focuses on the stage-1 problem that has a fast, fixed benchmark and can be optimized overnight by an agent.

The repo pattern comes from Karpathy's [autoresearch reference clone](/Volumes/bkawk/projects/mesh-para/reference), which we used as the starting point for the one-file autonomous research loop design.

@matthew_berman helped put `autoresearch` on our radar in the first place, which is what kicked off this direction for the project.

We also borrowed autonomy and workflow ideas from a few adjacent repos while shaping the manager layer:

- [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch): the core fixed-budget, fixed-metric, keep-or-revert research loop pattern
- [`obra/superpowers`](https://github.com/obra/superpowers): staged workflow, isolated workspaces, and strategy-pack thinking
- [`garrytan/gstack`](https://github.com/garrytan/gstack): role separation, orchestration, and manager-style control-plane ideas

This repo stays much narrower than those systems. We borrowed useful patterns, not their full product or agent framework scope.

## Why this exists

Mesh -> STEP is not a normal format conversion. The mesh has already discarded design intent, and recovering CAD means inferring:

- surfaces
- boundaries
- primitive parameters
- eventually topology and valid BRep structure

This repo takes the `autoresearch` loop idea and applies it to that first learnable part.

## Current status

The project has moved past the synthetic scaffold stage:

- real ABC-backed datasets are working
- a guarded autonomous loop is running
- the current best improvements came from better supervision, not bigger models

Important milestones so far:

- weak PointNet-style baseline on real ABC subset:
  - `val_score 0.137444`
- first strong local-geometry model:
  - `val_score 0.329439`
- boundary-supervision baseline:
  - `val_score 0.332057`
- current overnight loop best:
  - `val_score 0.359468`

## Repo contract

The repo follows the same core contract as the cloned [autoresearch reference repo](/Volumes/bkawk/projects/mesh-para/reference):

- [build_dataset.py](/Volumes/bkawk/projects/mesh-para/build_dataset.py): builds or packs fixed shards
- [prepare.py](/Volumes/bkawk/projects/mesh-para/prepare.py): read-only infrastructure, loading, evaluation
- [train.py](/Volumes/bkawk/projects/mesh-para/train.py): the main editable research file
- [program.md](/Volumes/bkawk/projects/mesh-para/program.md): human-written research brief
- [research.py](/Volumes/bkawk/projects/mesh-para/research.py): autonomous keep-or-revert loop

The benchmark is intentionally simple:

```text
val_score = 0.7 * macro_iou + 0.3 * (1 / (1 + param_rmse_norm))
```

Higher is better.

## Docs

More detailed documentation lives in:

- [Project Overview](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/overview.md)
- [Findings](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/findings.md)
- [Workflow](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/workflow.md)
- [Next Phase](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/next_phase.md)
- [Shared Lab Memory](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/shared_lab_memory.md)
- [Contributing](/Volumes/bkawk/projects/mesh-para/cadresearch/CONTRIBUTING.md)

## Quick start

```bash
cd /Volumes/bkawk/projects/mesh-para/cadresearch
uv sync
uv run build_dataset.py synthetic
uv run prepare.py
uv run train.py
```

For real data, pack preprocessed `.npz` samples instead of using the synthetic bootstrap.

## Running the loop

Seed the autonomous loop from an existing good run:

```bash
python3 research.py seed \
  --run-name boundary512 \
  --baseline-log artifacts/abc_boundary_512.log
```

Run one guarded iteration locally:

```bash
python3 research.py loop \
  --run-name boundary512 \
  --iterations 1 \
  --train-command 'MESH_PARA_CACHE_DIR=artifacts/abc_cache_512_boundary python3 train.py'
```

Or run training on `bkawk.local`:

```bash
python3 research.py loop \
  --run-name boundary512 \
  --iterations 1 \
  --pre-train-command 'scp train.py bkawk.local:/data/projects/mesh-para/train.py' \
  --train-command "ssh bkawk.local 'cd /data/projects/mesh-para/cadresearch && MESH_PARA_CACHE_DIR=/data/projects/mesh-para/artifacts/abc_cache_512_boundary PYTHONUNBUFFERED=1 python3 train.py'"
```

Check the current best:

```bash
python3 research.py status --run-name boundary512
```

By default, `research.py` now reruns any apparent winner once before promotion and records close sub-threshold results as `near_miss` instead of treating them as ordinary reverts.

## Managing autonomy

`research.py` is the worker loop. [manager.py](/Volumes/bkawk/projects/mesh-para/cadresearch/manager.py) is the lab-manager layer that decides what to try next when a run plateaus.

Ask the manager what it thinks should happen next:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  recommend --source-run boundary512_refocused
```

By default, `recommend` stays cache-only while any `research.py loop` is active, so check-ins do not kick off fresh 2048-set audits and steal GPU time from the live worker. Missing audits are queued in [artifacts/manager/default/audit_queue.json](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/audit_queue.json) for the idle supervisor to process later, with higher-scoring unresolved branches prioritized first. If you explicitly want a heavier refresh while the lab is busy, pass `--allow-live-audits`.

For a side-effect-free dashboard view of the current lab state:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  status --source-run boundary512_refocused
```

To fingerprint the remote GPU and cache a hardware profile for future manager decisions:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  hardware
```

That writes a cached profile under [artifacts/manager/default/hardware](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/hardware). If the cached profile says the hardware tier is `expanded` and [strategies_expanded](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies_expanded) exists, the manager automatically prefers that expanded strategy pack instead of the default [strategies](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies).

The status view now distinguishes between the current raw-score leader and the currently resolved canonical baseline. If those differ, the gap usually means a stronger branch exists but has not yet cleared the audit-promotion gate. The `audit_pending` field tells you whether the current leader is already queued for that audit.

Have it plan the next run without launching:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  launch-next --source-run boundary512_refocused --dry-run
```

Or let it promote the best baseline from an existing run and launch the next strategy automatically:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  launch-next --source-run boundary512_refocused --require-idle
```

When the manager has prior launched runs in its history, it automatically resolves the highest-scoring completed run and uses that as the next baseline. That means you can keep pointing it at the same original source run and it will promote forward as soon as a manager-launched branch beats the old baseline.

To keep the lab moving with no manual relaunches, run the supervisor layer:

```bash
python3 manager.py --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  supervise --source-run boundary512_refocused --poll-seconds 600
```

For a more reliable single-instance background launcher, use [supervisor_ctl.sh](/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh):

```bash
/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh start
/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh status
/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh restart
/Volumes/bkawk/projects/mesh-para/cadresearch/supervisor_ctl.sh stop
```

The supervisor checks whether any `research.py loop` process is active. If the lab is idle, it asks the manager to launch the next run from the best completed baseline it knows about.

The supervisor log at [artifacts/manager/default/logs/autonomy.log](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/logs/autonomy.log) now includes compact status snapshots during active runs, and richer decision lines when the lab is idle enough to audit or launch, so you can read the manager's reasoning as a timeline.

Manager-launched runs execute from disposable copied workspaces under [artifacts/manager](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager), so the main repo checkout can stay clean while autonomy is running.

Strategy families now live in [strategies](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies) as small JSON packs, and completed manager-launched runs get markdown retros under [artifacts/manager/default/retros](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/retros). The manager now uses those retros to tag strategies as `continue`, `probe_nearby`, `reliability`, or `cool_down` instead of relying only on raw score totals.

Manager promotion decisions now also use cached 2048-set audit reruns under [artifacts/manager/default/audits](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/audits). Those audits run in isolated remote workspaces so they do not interfere with the main overnight worker. The first recommendation after enabling this can be slower because the manager may need to backfill audit results for older candidate runs.

## Working with ABC

The current real-data path uses matched ABC mesh, STEP, and feature archives:

1. extract a matched subset with [extract_abc_subset.py](/Volumes/bkawk/projects/mesh-para/extract_abc_subset.py)
2. preprocess raw samples with [preprocess_abc_raw.py](/Volumes/bkawk/projects/mesh-para/preprocess_abc_raw.py)
3. pack them into fixed shards with [build_dataset.py](/Volumes/bkawk/projects/mesh-para/build_dataset.py)

The extracted raw subset contains:

- `mesh.stl`
- `model.step`
- `features.yml`

The packed sample format currently includes:

- `points`
- `normals`
- `labels`
- `params`
- `param_mask`
- `boundary`

## Roadmap

Near-term:

- keep improving stage-1 geometric understanding
- use the overnight loop to search around the new boundary-supervision baseline
- validate wins on secondary audit settings

## Hardware note

For this repo, a faster GPU would matter a lot.

Because the benchmark is a fixed wall-clock run, better hardware does not just make experiments finish sooner. It can change which models win:

- more steps inside the same 5-minute budget
- more room for slightly richer local-geometry models
- better overnight throughput for the autonomous loop

In practice, a single RTX 5090 would likely be one of the highest-leverage upgrades for this project. Two 5090s would help even more if used as parallel experiment workers rather than as a distributed trainer.

The repo now has a built-in prep path for that upgrade: cache the new GPU profile with `manager.py hardware`, then let the manager switch into the expanded strategy pack automatically. That gives the lab a controlled way to revisit slightly richer local models and wider neighborhoods when the hardware actually supports them.

More discussion is in [Findings](/Volumes/bkawk/projects/mesh-para/docs/findings.md).

Later:

1. primitive instance grouping
2. primitive fitting and trimming
3. topology assembly
4. valid BRep / STEP export
