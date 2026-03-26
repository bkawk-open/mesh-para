# cadresearch

This is an `autoresearch`-style loop for mesh-to-CAD surface understanding.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag based on today's date, for example `mar26`.
2. Create a fresh branch `cadresearch/<tag>`.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
4. Verify the dataset exists:
   - If not, run `uv run build_dataset.py synthetic` for a smoke test dataset.
   - For real data, the user should first pack prepared `.npz` samples with `uv run build_dataset.py pack --input-dir ...`.
5. Initialize `results.tsv` with the header row only.
6. If `research.py` is present, prefer using it as the experiment harness.
7. Confirm setup looks good.

## Experimentation

Each experiment runs for a fixed wall-clock budget. The training script prints a single scalar `val_score` plus supporting metrics. The goal is to maximize `val_score`.

You may modify:

- `train.py` only

You may not modify:

- `prepare.py`
- `build_dataset.py`
- the validation data
- the metric

## What matters

- Normals are available and informative.
- Mechanical parts are dominated by planes and cylinders, but macro-IoU means rare classes still matter.
- Local geometry should matter more than pure global pooling.
- Simpler changes are better when gains are similar.

## First run

The first run should always establish the baseline:

```bash
uv run train.py
```

## Output format

Each run ends with a summary like:

```text
---
val_score:        0.512345
macro_iou:        0.478901
param_rmse_norm:  0.812345
param_score:      0.551234
training_seconds: 300.0
peak_vram_mb:     1234
---
```

Record results in `results.tsv` as:

```text
commit	val_score	memory_gb	status	description
abc1234	0.512345	1.2	keep	baseline
```

Use `0.000000` and `0.0` for crashes.

## Research directions

The baseline is intentionally weak:

- it ignores normals in the encoder
- it uses no local neighborhoods
- it uses a single global max pool
- it predicts primitive parameters with a shallow head

Good directions include:

- normal-aware feature extraction
- k-NN or radius-graph local aggregation
- multi-scale PointNet++ style grouping
- transformer blocks over local neighborhoods
- better class balancing or focal-style losses
- parameter heads that are class-aware

## Loop

1. Look at the current git state.
2. Modify `train.py` with one experimental idea.
3. Commit the change.
4. Run `uv run train.py > run.log 2>&1`.
5. Read the summary from `run.log`.
6. Record the result in `results.tsv`.
7. Keep the commit only if `val_score` improves meaningfully.

When using `research.py`, let the harness manage keep-or-revert decisions automatically.
