# 1267_rd07_q6k_mmq_scale_fold

**Status:** untested
**Plan item:** PRBE110

## What it does

Hoists and folds Q6_K MMQ scales and emits a one-shot activation marker from the composed forced-J dispatch path.

## Why

RD07 is isolated from rejected bundle 1203 and requires fresh validation under its own identity.

## Upstream

Adapted from stew675 RD07. Requires `0300_mmq_forced_j` because the activation edit anchors on its composed Q6_K dispatch signature.
