# 1255_nro06_adaptive_mtp_depth

**Status:** rejected
**Plan item:** NRO06

## What it does

Adds a pure adaptive MTP depth controller beside the existing MTP implementation. It is inert unless runtime wiring 1268 is composed and enabled.

Current policy: reset at depth 3 clamped to floor/cap; evaluate a 32-drafted-token window; <=60% acceptance drops one depth, >=72% climbs one depth, otherwise hold. State is per sequence and all transitions are integer functions of accepted/drafted counts.

## Why

The prior floor-start/climb/drop policy made floors 1/2 a cold-start penalty and could not distinguish the observed short-context loss from the long-context gain. The new hysteresis keeps the controller deterministic while allowing sustained low acceptance to move from 3 to 2.

## Upstream

Local staged adaptation of nasone commit `10579a7365a3bc86c4f8e41aaab20e73e1571e5e`; the 2026-10-08 hysteresis is BigCherry-local and pending hardware qualification.

## Result at b11474: rejected (2026-10-09)

The controller is only reachable through 1268; see `patches/1268_prbe52_adaptive_mtp_wiring/SUMMARY.md` for the measurements. Every adaptive configuration tested was slower than the fixed draft depth on Flash-Next and on the 27B.
