---
id: BPB01
order: 0
plan: build-pin-bump
state: pending
created-at: '2026-10-07T14:22:02.591382+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P2
work: M
---

# b11474: reconcile the non-production patches that no longer apply

## Description

The b11402 -> b11474 bump (2026-10-08) reconciled the production set (48 patches compose; 1343, 1335 and 1336 retired as superseded by upstream 1a3011cc0, #29901 and #29943; 0200, 1292, 1326, 1327 and 1341 re-anchored). Thirteen patches outside the production set do not apply at b9acf138 and carry a known_broken disposition bound to that revision until each is either reconciled or retired: 1210_rd26_bitidentical_decode_verify_standalone (two mmvf decode-verify anchors), 1268_prbe52_adaptive_mtp_wiring (blocked by 1210), 1250_nro01_allreduce_q8_wire, 1275_ar_small_latency, 1293_sched_single_input_sync (scheduler loop extracted upstream), 1320_meta_compute_timing, 1321_mtp_ahead_primitives, 1322_mtp_ahead_overlap (blocked by 1321), 1328_aux_rocm_expert_backend (scheduler copy), 1337_moe_expert_caching and 1338_moe_cache_profile (built on 1336's callback, now upstream's: MET07), 1342_fusion_bisect (FKE01; upstream e117148a4 changed the alloc_deps check), 1346_mtp_prompt_overlap (QFP31; upstream f0c41e016 consolidated the nextn row cropping).

## Steps

1. 1346 with the QFP31 work (re-brief GPT against the new pin). 2. 1342 with FKE01, after reading upstream e117148a4. 3. 1320 (needed by the prefill-diag experiment). 4. 1337/1338 on upstream's copy callback (MET07). 5. Decide retire-or-fix for the rest; clear each disposition as it is reconciled.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-rebase-check --all reports every non-rejected patch clean or retired; patch-disposition list is empty for b9acf138.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

- 1322_mtp_ahead_overlap — applies at b11474 in experiment.deploy-v6-plus-ahead once required 1321 is present; no 1322 anchor changes needed.

- 1321_mtp_ahead_primitives — re-based for b11474: forced-front/live-tail logic is unchanged; only the MTP n_min post-pass anchor moved with upstream nextn restructuring.

- 1338_moe_cache_profile — applies after rebased 1337 at b11474; no cache-source anchors changed. Test now builds from the current pin and no longer carries retired 1336.

- 1337_moe_expert_caching — re-based for b11474: dropped retired 1336 dependency, retained upstream #29943 selective copy callback, and prepares cache-owned host expert weights in the host-weight pass before delegating ordinary weights to ggml_backend_sched_copy_input.

- 1328_aux_rocm_expert_backend — re-based for b11474: moved the pinned Meta/aux staging hook into upstream's extracted ggml_backend_sched_copy_input fallback; all other anchors still match b11474.

- 1293_sched_single_input_sync — re-based for b11474: carry the once-per-split user-input sync state through upstream's extracted ggml_backend_sched_copy_input helper; production 1326 still composes after it.

## Change Log

- 2026-10-07T14:22:02.591382+00:00 (created-by): Created by agent

## Ledger-events




- chg_20261007_155512_bigcherry-now-builds-on-llama_6400
- 2026-10-07T15:55:15.944121+00:00 (updated-by): Updated: section:ledger-events
- chg_20261008_015302_faster-prompt-processing-6-10_5913
- 2026-10-08T01:53:16.014253+00:00 (updated-by): Updated: section:ledger-events
- chg_20261008_015330_faster-prompt-processing-6-10_3634
- 2026-10-08T01:53:44.227179+00:00 (updated-by): Updated: section:ledger-events
- chg_20261008_015344_diagnostic-traces-available-in_3686
- 2026-10-08T01:53:50.935436+00:00 (updated-by): Updated: section:ledger-events
