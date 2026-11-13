# Unpublished branch reconciliation — RSC11 P1 and RHP19

Date: 2026-11-13. Scope: source/control-plane review and card-free testing only. **No GPU lease, hardware measurement, inference run or production change was performed.** The two original commits had clean local working trees, each one commit ahead of its old base and absent from `main` by patch identity. Their complete content has now been preserved remotely; **neither prototype has been applied to production**.

## Archival provenance (source trees preserved byte-for-byte)

| Programme | Original unpublished local commit | Git tree (original = remote archive) | Permanent remote archive | Disposition |
| --- | --- | --- | --- | --- |
| RSC11 S13 P1 | `006b7ce149c4b5f943acb6552c202a9bf2648f0c` | `7256e9879a7ba32626baa99cc075b7da4f44ace7` | [`archive/rsc11-s13-p1-engine-native-20261113`](https://github.com/audiagentic/bigcherry/tree/archive/rsc11-s13-p1-engine-native-20261113), archive commit `e4f6c4bd79c153afd25d01a096b35afb3f134469` | Superseded; do not merge `rad0014` |
| RHP19 drift envelope | `1c486f28a1cc31ff7ebc1ad9c283edf8c77f7c16` | `a40d6def91720be616f6db80a4c42388a6c2f987` | [`archive/rhp19-drift-envelope-20261113`](https://github.com/audiagentic/bigcherry/tree/archive/rhp19-drift-envelope-20261113), archive commit `e76e57daf63f7955d6b15cbddc91785213ca42c7` | Superseded implementation; retain findings |

The archive commits were recreated on the *original parent commits* (`2bac0a87ba87df6f53a5de5e62aa515a5c7e7340` and `25e19de091c91c47bab8761353c991b3afb4564c`, respectively) using the original complete file contents. Git tree SHA equality was verified, not merely similarity of patch text. The remote archive commit IDs differ from the original local commits because the archive commits have new commit metadata; **the tracked source tree is identical**.

## RSC11 P1: why the inert `rad0014` prototype is retired

Original two-file patch: `engines/radiance/patches/rad0014_plan_memory_v2_inputs/{patch.py,patch.toml}` in the archived branch. It creates a second `--plan-memory-v2-inputs FILE` engine flag, separate header/CPP, an extra per-rank ledger and a second sidecar. Its `required_bytes` is computed from `rank_cfg[i].vram_weights_mib + vram_kv_mib`; its `fixed_bytes` is only `facts[i].statics`. Its JSON `gpu` elements have `bdf,required_bytes,fixed_bytes`, without `claimable_bytes` or `evidence`, and the sidecar kind is `radiance_engine_gpu_inputs_v2`.

Current `main` already owns this boundary inside **`engines/radiance/patches/rad0011_plan_memory/patch.py`**. The `--plan-memory-ranks` publication is connected to final `vram_budget_rank_split` inputs, includes arena, staging, queue and statics overhead, checks bounded integer arithmetic, and emits per-rank BDF, required, fixed, claimable and evidence fields. **`engines/radiance/components/serving/v2/p1_producer.cpp`** requires exactly that schema (`schema_version=1`, `kind=radiance_p1_rank_demand`, `resident_other_bytes`, all five GPU members), plus a strict provenance string. The archived `rad0014` output is incompatible, so integrating it as another publisher would weaken authoritative measurement and add a parallel interface. Its `patch.toml` explicitly says `state = "untested"`; there are no separately committed patch tests for this prototype.

**Decision:** keep the present `rad0011` plus native P1 producer. Do not apply the old `rad0014`, register an additional command, or infer that the archived prototype ever passed current-pin or hardware gates. RSC11's active completion/admission gates remain governed by the existing RSC11 plan and current code, not by the archive.

## RHP19: what the unmerged fix taught us and what replaced it

Original six changed files: `RHP18.md`, `RHP19.md`, `worker_policy_driver.h`, `calibration_watchdog.py`, `launch-flashnext-best.sh` and `test_rhp19_calibration_watchdog.py`. The branch diagnosed the **240-second** launcher cutoff even though the selected A/B/B/A workload could contain up to 120 long requests, plus the adaptive drift ring's measured capacity/horizon. It also documented a host-pool-key split (`engine.host_pool_mib` 9216 -> 2259) and correctly cautioned that an un-ingested helper marker is *not* an accepted calibration. These findings are retained as historical evidence, not as a new hardware acceptance result.

The archived implementation emits a plain `RHP19_CALIBRATION_PROGRESS` line from `worker_policy_driver.h`, parses its numeric fields, and widens the deadline using policy, workload and drift. Current `main` instead implements the stronger structured `RHP19_DRIFT_PROGRESS_JSON` contract, `measured_drift`, `workload_budget` and `bounded_deadline`, with a maximum two-workload-traversal envelope and explicit no-feed refusal, through merged [#828](https://github.com/audiagentic/bigcherry/pull/828), [#833](https://github.com/audiagentic/bigcherry/pull/833) and [#836](https://github.com/audiagentic/bigcherry/pull/836). Reverting to the archived free-text telemetry or its earlier bound would regress the current protocol and its tests.

**Card-free evidence at reconciliation:** `python3 -B tools/tests/lab/test_rhp19_calibration_watchdog.py` on `main`: **15/15 pass**; same script on the archived original branch: **12/12 pass**. Python's `pytest` module was not installed in the available runtime, so these are direct stdlib `unittest` results; no claim of full CI or live hardware acceptance. The original branch documentation described `runs/rhp18-acceptance/calib.log` and `runs/recalibrate-4gpu`; these were **not re-executed or revalidated** by this reconciliation. They remain provenance pointers, not newly retained raw receipts.

**Decision:** do not cherry-pick the older RHP19 code over `main`. Retain historical findings above and the full archive. **RHP19 remains in progress:** a lease-held four-card calibration must produce an accepted record under the current key, followed by a plain policy-ON start; the missing-feed and structural/no-crossing refusal cases and other layouts still require explicit gate evidence. The earlier failure and its analysis are not a production-ready result.

## Source-control cleanup and guardrails

Both local branches originated on old main commits and were not valid fast-forward integration candidates. After archive tree equality and this managed evidence record, their temporary worktrees and local branch refs may be removed. Archive branches are intentionally retained on GitHub so the complete old source cannot become local-machine-only evidence. This reconciliation does not close RSC11/RHP19 work items, does not activate either abandoned protocol and does not claim any GPU validation.
