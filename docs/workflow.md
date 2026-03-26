# Workflow

## Local development

Basic smoke-test flow:

```bash
cd /Volumes/bkawk/projects/mesh-para/cadresearch
uv sync
uv run build_dataset.py synthetic
uv run prepare.py
uv run train.py
```

## Real-data workflow

Current real-data flow:

1. Extract a matched raw ABC subset.
2. Preprocess raw samples into per-object `.npz`.
3. Pack them into fixed train/val shards.
4. Point `MESH_PARA_CACHE_DIR` at the packed shard cache.
5. Run `train.py`.

## Remote training

This project often edits locally and trains on `bkawk.local`.

Typical pattern:

```bash
scp /Volumes/bkawk/projects/mesh-para/train.py bkawk.local:/data/projects/mesh-para/train.py
ssh bkawk.local 'cd /data/projects/mesh-para/cadresearch && MESH_PARA_CACHE_DIR=/data/projects/mesh-para/artifacts/abc_cache_512_boundary PYTHONUNBUFFERED=1 python3 train.py'
```

## Autonomous loop

Seed from a known good run:

```bash
python3 /Volumes/bkawk/projects/mesh-para/research.py seed \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  --run-name boundary512 \
  --baseline-log /Volumes/bkawk/projects/mesh-para/artifacts/abc_boundary_512.log
```

Run one guarded iteration:

```bash
python3 /Volumes/bkawk/projects/mesh-para/research.py loop \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  --run-name boundary512 \
  --iterations 1 \
  --agent codex \
  --pre-train-command 'scp train.py bkawk.local:/data/projects/mesh-para/train.py' \
  --train-command "ssh bkawk.local 'cd /data/projects/mesh-para/cadresearch && MESH_PARA_CACHE_DIR=/data/projects/mesh-para/artifacts/abc_cache_512_boundary PYTHONUNBUFFERED=1 python3 train.py'"
```

Check status:

```bash
python3 /Volumes/bkawk/projects/mesh-para/research.py status \
  --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch \
  --run-name boundary512
```

## Overnight run

There is a helper script:

- [overnight_boundary512.sh](/Volumes/bkawk/projects/mesh-para/overnight_boundary512.sh)

It is currently being run inside a detached `screen` session for overnight experimentation.

Useful commands:

```bash
screen -ls
tail -f /Volumes/bkawk/projects/mesh-para/artifacts/autoresearch/boundary512/overnight_loop.log
python3 /Volumes/bkawk/projects/mesh-para/research.py status --project-dir /Volumes/bkawk/projects/mesh-para/cadresearch --run-name boundary512
```

Stop the detached run:

```bash
screen -S boundary512_overnight -X quit
```
