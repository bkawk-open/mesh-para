# Shared Lab Memory

This repo is trying to act like a small open research lab, not just a code dump.

That means contributors should be able to learn from our:

- keepers
- regressions
- near misses
- supervision changes
- hardware constraints

and we should be able to learn from theirs the same way.

## What counts as shared memory

The most important research memory in this repo lives in:

- [README.md](/Volumes/bkawk/projects/mesh-para/cadresearch/README.md)
- [docs/findings.md](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/findings.md)
- [docs/next_phase.md](/Volumes/bkawk/projects/mesh-para/cadresearch/docs/next_phase.md)
- [strategies](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies)
- [strategies_expanded](/Volumes/bkawk/projects/mesh-para/cadresearch/strategies_expanded)
- manager retros under [artifacts/manager/default/retros](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/retros)
- run histories under [artifacts/autoresearch](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/autoresearch)
- cached audits under [artifacts/manager/default/audits](/Volumes/bkawk/projects/mesh-para/cadresearch/artifacts/manager/default/audits)

## What we want contributors to preserve

When a contributor finds something interesting, the useful payload is not just the patch.

We want:

- the patch
- the measured outcome
- the hardware context
- the reasoning
- whether it should be explored more or retired

That applies to:

- winners
- almost-winners
- clean failures

## Why this matters

The project is unusually sensitive to:

- fixed wall-clock budget
- hardware throughput
- search-space design
- supervision quality

Because of that, a result without context is not very transferable.

The same code change can mean different things on:

- an RTX 4060
- a 5090
- two GPUs running parallel branches

Shared memory is how we keep those results interpretable.

## Ideal contribution record

For a useful external result, we want to know:

- what baseline it started from
- what exact change was made
- main 512-set benchmark result
- 2048-set audit result, if available
- hardware fingerprint
- whether it looks like:
  - a keeper
  - a near miss
  - a dead end

## Desired long-term direction

The long-term goal is that this repo becomes a place where:

- our misses save others time
- their wins move us forward
- different hardware tiers explore different parts of the search space
- strategy packs and docs carry the research program forward

That is the open-source version of the collaborative autonomous-research idea that inspired this project in the first place.
