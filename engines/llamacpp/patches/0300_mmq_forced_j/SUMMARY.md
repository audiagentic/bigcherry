# 0300_mmq_forced_j: MMQ forced-J variant dispatch (HI06)

**Status:** validated
**Plan item:** none

## What it does

Splits upstream's mul_mat_q_switch_J into a scan (mul_mat_q_compute_J_best, lifted unchanged) and a launcher (mul_mat_q_launch_forced_J) that takes J as an explicit parameter, so a forced value can override the scan's answer while the native path stays identical.

## Why

The tuner needs to select and measure a specific MMQ tile width J instead of only ever seeing upstream's own scanned choice; separating the scan from the switch is the least invasive way to do that since launch_mul_mat_q already templates on J.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI06).

## PA44-B owned definitions

The two whole file-scope blocks now live in BigCherry-owned includes:
`ggml/src/ggml-cuda/bc-mmq-forced-j.cuh` for the MMQ eligibility helper/declaration and
`ggml/src/ggml-cuda/bc-mmq-forced-j-source.cuh` for the tuner/variant definitions. They are included at the
former definition sites so file-local MMQ helpers/types remain available. Public signature edits, forced-J
selection, callers and autotune hooks are unchanged.

## Identity proof

The moved blocks contain 101 physical lines / 94 non-blank whitespace-stripped lines,
preserved exactly in the two owned files. Replacing the two include lines with those files reconstructs the
pre-refactor composed source; the only multiset delta is the two include lines (owned-definition multiset
94/94, include delta +2).
