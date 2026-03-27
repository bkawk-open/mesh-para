# Next Phase

The clearest next research phase for `cadresearch` is:

1. add primitive instance supervision
2. upgrade the manager to hierarchical strategy search

That gives us one improvement to the task itself and one improvement to how the autonomous lab explores it.

## Better Supervision

The strongest next supervision target is primitive instances.

What that means:

- each point keeps its surface type label such as `plane` or `cylinder`
- each point also gets an instance id saying which actual primitive it belongs to

Why this is likely the next big step:

- it bridges segmentation and fitting
- it makes downstream primitive fitting much easier
- it matches the real CAD problem better than per-point class labels alone

### Proposed implementation path

1. extend [preprocess_abc_raw.py](/Volumes/bkawk/projects/mesh-para/cadresearch/preprocess_abc_raw.py) to emit instance or group labels from ABC features
2. extend [build_dataset.py](/Volumes/bkawk/projects/mesh-para/cadresearch/build_dataset.py) and [prepare.py](/Volumes/bkawk/projects/mesh-para/cadresearch/prepare.py) to carry those labels through the packed shard format
3. add a lightweight instance-aware objective in [train.py](/Volumes/bkawk/projects/mesh-para/cadresearch/train.py)

That objective should stay small and 5-minute-budget friendly. The first pass should likely be:

- pairwise same-instance affinity prediction, or
- a simple embedding loss for same-primitive grouping

This phase should not jump all the way to full clustering or topology reconstruction inside the benchmark loop.

## Smarter Strategy Search

The next manager upgrade should be hierarchical search.

Instead of choosing only among flat strategy names like:

- `boundary_calibration`
- `class_fusion`
- `global_context`

the manager should choose:

- a strategy family
- then a variant inside that family

Example:

- family: `instance_learning`
- variants:
  - `pairwise_affinity`
  - `embedding_margin`
  - `boundary_instance_coupling`

### Why this matters

- the search space is becoming more structured
- the manager should reason over research directions, not only individual prompts
- near misses should cause local exploration, not random family switching

### Proposed implementation path

1. keep current strategy packs
2. add metadata such as:
   - `family`
   - `hardware_tier`
   - `depends_on`
   - `explore_after`
3. teach [manager.py](/Volumes/bkawk/projects/mesh-para/cadresearch/manager.py) to:
   - choose a family first
   - choose a member strategy second
   - stay near recent near-misses when appropriate
   - cool down entire weak families after repeated failures

## Recommended Order

1. let the current autonomy run finish or plateau
2. add instance labels to the dataset pipeline
3. add one minimal instance-aware loss in [train.py](/Volumes/bkawk/projects/mesh-para/cadresearch/train.py)
4. introduce a new `instance_*` strategy family
5. then upgrade the manager to family-aware routing

## What Not To Do Yet

- do not jump straight to full BRep or topology supervision
- do not add a huge new model family at the same time as new supervision
- do not broaden the autonomy search before the instance supervision path exists

## Summary

The practical roadmap is:

- next scientific step: primitive instance supervision
- next autonomy step: family-aware search
- later: use stronger hardware to revisit richer local geometry inside that better-structured system
