---
id: MET07
order: 7
plan: patching-moe-expert-tiering
state: pending
created-at: '2026-10-06T14:52:56.714383+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# Profile-pinned expert cache (1337 + 1338): qualify for single-card and over-VRAM use

## Description

Per-expert residency for host-resident experts (Strata's idea on upstream's cache): every layer's experts on the host, a GPU cache with a frequency-profile pinned hot set and an LRU tail, large prefill batches through the cache. Results so far are on the Swift IQ2_XS file on one R9700 only (recorded in MET01): LRU cache +60% decode over whole layers at similar VRAM; pinned profile lifts prefill from 130 to 350-440 t/s; decode holds only when the profile covers the workload.

## Steps

1. Record a prefill + decode profile on UD-IQ4_XS (ARMS=record exists; moecopy-b11402y-r9700/profile.bin is a first one) and run the profile lanes on UD (W / P / PS / pinned-share sweep). 2. Same on one RX 7900 XTX (gfx1100). 3. With the MTP drafter on the 6900 XT (earlier whole-layer comparison with MTP went against the cache). 4. With -ub 2048 / 8192 (whole layers gained 2.5-4x prefill from the larger micro-batch). 5. Fidelity probes for the pinned arms (open: at 85% pinned the short request's text differs from the LRU arm; suspected fallback of a small slot group to the non-cache path, unverified). 6. Decide: promote 1337 + 1338 as opt-in for single-card / over-VRAM deployments, or reject.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

patches/1337_moe_expert_caching, patches/1338_moe_cache_profile, tools/lab/flash-next/moe-copy-ab.sh, queue-moe-copy.sh, tools/lab/strata/routing-skew.py

## Validation

Lightweight tier: offline tests + lint, activation marker (BIGCHERRY_PATCH_HIT patch=1338), ABBA against whole layers at equal VRAM with complete separation on prefill and no decode loss with an in-domain profile, fidelity inside the envelope, on gfx1201 and gfx1100.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Scope limit: the cache refuses the tensor split and multiple devices, so this does not apply to the 3-GPU production layout, where everything fits in VRAM. Value: single-card use and higher quants.

## Change Log

- 2026-10-06T14:52:56.714383+00:00 (created-by): Created by agent
