# 0300_mmq_forced_j: MMQ forced-J variant dispatch (HI06)

**Status:** validated
**Group:** core
**Plan item:** none

## What it does

Splits upstream's mul_mat_q_switch_J into a scan (mul_mat_q_compute_J_best, lifted unchanged) and a launcher (mul_mat_q_launch_forced_J) that takes J as an explicit parameter, so a forced value can override the scan's answer while the native path stays identical.

## Why

The tuner needs to select and measure a specific MMQ tile width J instead of only ever seeing upstream's own scanned choice; separating the scan from the switch is the least invasive way to do that since launch_mul_mat_q already templates on J.

## Upstream / provenance

Local design, part of this project's own HIP measured-dispatch framework (HI06).
