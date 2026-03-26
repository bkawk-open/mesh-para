#!/bin/zsh
set -euo pipefail

cd /Volumes/bkawk/projects/mesh-para/cadresearch

python3 research.py loop \
  --run-name boundary512_refocused \
  --iterations 60 \
  --agent codex \
  --agent-prompt-extra 'Build on the current best boundary-supervision baseline. Make exactly one focused change in train.py. Prefer small changes to boundary-head usage, global-context mixing, or light class-specific feature fusion. Avoid deeper local stacks, second neighborhood passes, or larger k-NN neighborhoods. Protect throughput first and only keep changes likely to improve macro-IoU or param_score.' \
  --pre-train-command 'scp train.py bkawk.local:/data/projects/mesh-para/cadresearch/train.py' \
  --train-command "ssh bkawk.local 'cd /data/projects/mesh-para/cadresearch && CADRESEARCH_CACHE_DIR=/data/projects/mesh-para/cadresearch/artifacts/abc_cache_512_boundary PYTHONUNBUFFERED=1 python3 train.py'" \
  --baseline-log /Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/autoresearch/boundary512/logs/iter_0002_train.log \
  --min-improvement 0.001
