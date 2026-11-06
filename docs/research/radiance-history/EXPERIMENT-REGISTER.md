# Radiance experiment register

One row per experiment that did not ship as a default, with outcome and reason. Add a row when an experiment branch
is closed or deleted. The detailed notes are linked; the code lives on the named branch or tag until removed.

| Experiment | Item | Outcome | Why | Detail |
|---|---|---|---|---|
| Blocking lanes, GPU-side arrival waits | RAD20 | Rejected | About -14% decode; any pending GPU wait burns about 80 W per XTX (MES fw 0x80). Replaced by burst gating. | `RAD20-helper-waits-2026-10.md`; branch `feat/rad20-blocking-lanes` |
| Large-lane sleep on host callback | RAD26 | Failed on hardware, open | Under HIP stream capture the callback never woke the helper; engine hung. Callback is 25-67 us p50 versus 6-8 us for the spin. | `docs/planning/active/patching-radiance-rdna3/RAD26.md`; branch `feat/rad26-large-lane-wake` |
| Wave-aggregated route compaction in mailbox publish | RAD29 | No result recorded | Branch holds code and an equivalence/timing test; item still pending. | branch `feat/rad29-route-compaction` |
| Helpers on efficiency cores, EPP settings | PWR03 | No result recorded | Harness `pwr03-ecore-ab.sh` written; host is Intel i7-12700KF. | `PWR03.md`; branch `feat/pwr03-cpu-placement` |
| Gated quantise bounded by route count (`LANE_QUANT`) | RAD23 | Shipped via #481 path; #482 closed as superseded | 616 to 28 us a request on the 6900 XT, replay hashes identical. | PR #482 |
| Multi-row packed-f16 W4 decode GEMV (`GEMV_W4MR`) | RAD23 | Closed unmerged (#506), default-off draft | Never validated end to end; needed its own A/B and KLD. | PR #506 |
| Rows-kernel accuracy variants (groupfold, outputlane, rb16, quad-DPP and others) | RAD46 | Quality fixed, speed not; resolved by row-count policy on `feat/rad46-accurate-rows` | See the note. | `RAD46-rows-kernel-accuracy-2026-11.md`; branches `codex/rad46*` |
| WSL06-A legacy inventory | WSL06 | Superseded | Work-in-progress path inventory; item completed another way. | branch `feat/wsl06a-legacy-inventory` |
| Platform release line from explicit change entries (`platform/changes/`) | release | Not adopted, archived | #564 `exclude-paths` fixed the 11552.0.0 platform version with less change; this stricter ownership model may be needed if unrelated changes keep triggering platform releases. | `experiments/release/PLATFORM-change-entries/` |


| 27B FP8/W8A8+DFlash2 and RDNA2 4/8 GiB tail layouts | 27B | Archived opt-in layouts | Three configuration-only commits; no measurements of exact variants, stopping reason unknown; no default changed. | `experiments/radiance/27B-fp8-w8a8-layouts/`; `exp/27b-fp8` |
| gfx1100 MXFP4 row policy | RAD47 | Shipped via #596 | Both branch changes on main (`619590728`); tiled from M=17; +27% at four streams, +37% at six. | `experiments/radiance/RAD47-xtx-rows/`; `exp/rad47-xtx-rows` |
| Helper serving-manifest original implementation | RSC06 | Superseded by #633; #527 closed | Original contract and helper-config patch replaced by v1 strict manifest reader. | `experiments/radiance/RSC06-helper-manifest/`; `feat/rsc06-helper-manifest` |


| Helper fp32 result planes, protocol v20 | RAD23 | Archived inert patch | Novel unrounded-f32 combine and CPU bench not validated end-to-end; later chunk-wise combine shipped (#497). | `experiments/radiance/RAD23-result-f32/`; `feat/rad23-result-f32` |
| Helper fp16 rows with exponent, protocol v21 | RAD23 | Archived inert patch | No recorded KLD or throughput A/B; later production protocol superseded the experiment. Apply only after v20 patch on historical base. | `experiments/radiance/RAD23-result-f16/`; `feat/rad23-result-f16` |


| 1357 router split-K with 1358 eviction, 1330 mask trial | QFP17 | Shipped/partly rejected via main PRs | 1357 and scripts already present; 1358 via #107; 1330 ub1024 not adopted at 245760 (#77). | `experiments/radiance/router-splitk-cache-evict/`; `exp/router-splitk-with-cache-evict` |
| Four-card Flash-Next with mixed heat default | RAD23 | Shipped via #488 | Launcher/layout and mixed heat already on main; +48% eight-session prefill in historical A/B. | `experiments/radiance/RAD23-flashnext-4gpu-default/`; `feat/flashnext-4gpu-default` |


| ROCm SDK-pinned Radiance build | RCS11 | Shipped via #524 | Selected SDK code, validation and plan already on main; no residual patch. | `experiments/radiance/RCS11-rocm-sdk-selection/`; `rcs11-merge` |
| Internal references and environment-driven paths | privacy | Shipped/superseded via #394, #399 and #544 | Older redaction substitutions obsolete; unsafe to replay historical removed strings. | `experiments/radiance/internal-refs-redaction/`; `redact-internal-refs` |

Branch dispositions are archived under `experiments/radiance/<ITEM>-<name>/`. Each has a README; only changes not on main have inert patches. Experimental-only layout entries may instead be copied to `engines/radiance/layouts.toml` without changing defaults. The original RAD46 notes and reviews are in `experiments/radiance/RAD46-rad46/`.
