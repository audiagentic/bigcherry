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



### Pin b11552 (2026-10-12): 1337 superseded, 1338 needs a rebuild

- b11552 carries upstream #29887 itself (`src/llama-moe-cache.cpp`, `--moe-cache-mib`, `LLAMA_ARG_MOE_CACHE_MIB`). 1337 (our backport of the pre-final PR) is marked `superseded`; its mechanics test and the experiments `moe-expert-cache` / `moe-cache-profile` are removed.
- 1338 was written on 1337's revision of the cache. Upstream's merged version is laid out differently: 5 of its 12 edits have no anchor (includes; resolve() now tests `n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots`; there is no `planned_ids` replay check or assign any more; `lru.plan(il, ids.data(), ids.size(), fills, n_hit)` lost the remapped-ids argument). This is a rebuild against upstream's slot_map design, not a re-anchor.
- Until then 1338 stays `untested` with a known_broken disposition bound to b11552, its test is skipped with that reason, and it must drop `requires = 1337` when rebuilt (base is upstream).
- First step of the rebuild: re-measure upstream's cache as shipped (LRU only) on the MET01 setup, since the +60% decode result was taken with our backport.

## Change Log

- 2026-10-06T14:52:56.714383+00:00 (created-by): Created by agent
- 2026-10-11T16:56:51.685546+00:00 (updated-by): Updated: section:notes
