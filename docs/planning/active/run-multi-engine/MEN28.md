---
id: MEN28
order: 0
plan: run-multi-engine
state: pending
created-at: '2026-11-05T07:37:48.187895+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P0
---

# Production radiance builds fail closed on stale patch evidence; re-qualify the shipped patches at the shipped pin

## Description

Found 2026-11-05 while answering the owner's question whether radiance still follows the llama.cpp testing method. The patch gates exist (bigcherry patch-gates --intent build, G1-G7, evidence qualification per pin) but the lane build path (campaign/build.py, lane.py, planner.py) never calls them: only the patch-gates command does. Result: both validated radiance patches (rad0002_moe_remote_handoff, rad0008_hc_upmix_rt_sweep) FAIL G4 at the current pin (rad0002's profile-evidence record is for v1.14.0, the pin is v1.16.0; rad0008 has no current qualifying record) and production builds, CI and the launcher build them anyway. pin-bump runs compile, plugin ABI, component and cpu-smoke gates but does not re-qualify patch evidence, so two bumps (1.14 to 1.16, then 1.17) passed without it. 2 of 9 radiance patches are validated against 70 of 151 for llama.cpp; the 7 untested include three in the production source (rad0005 host-compat, rad0009 serving-manifest, rad0011 plan-memory), declared so by required-state. Evidence and A/B measurements of this period were produced by lab scripts and ad hoc run scripts, not ab-benchmark or campaigns (MEN14 gap).

## Steps

1. After the v1.17.0 bump merges (PR #627), re-qualify at the shipped pin through bigcherry commands only: rad0002 (profile-evidence: interleaved ab-benchmark helper-on vs engine-local control, activation from helper logs, KL identity against a fresh reference, mechanics) and rad0008 (re-bind or re-record), each as recorded run.json under the patch's evidence, using the bigcherry-patch-qualification skill. Record draft acceptance and prompt digests (accept-bench, stream-bench prompt_sha256) with every decode comparison.
2. Wire the build gate: a lane whose source includes a patch set at the default required-state must pass patch-gates G4 (evidence qualifies at the current pin) for every patch in it, failing closed before materialisation; sets declared required-state = untested are exempt by that declaration and the build log says so. Land it only once step 1 has recorded evidence, so main never goes red. No flag, no grandfathering.
3. pin-bump: a reaudit phase after the compile gate runs patch-verify-evidence for the production sets and stops with the stale patches named and the exact qualification commands to run; the bump's plan names re-qualification as its last mandatory step before pin-release.
4. Decide, per patch, what the untested production members do: rad0009 and rad0011 are validated by the RSC06/A4 hardware manifest gate (bigcherry-2); rad0005 cannot compile its FFmpeg path on this host: keep it out of the release build or qualify it on a host that can.
5. The release record (releases/<pin>.json) carries the gate result and the evidence digests; MEN27 attaches the bundle only for a build that passed.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

patch-gates --intent build passes for every patch in the production sets at the shipped pin; a build with a stale or missing record fails before materialisation with the patch and reason named (offline test); pin-bump stops on stale evidence (offline test); the v1.17.0 release record carries the gate result.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner direction 2026-11-05: 'ensure we are using the build system - campaigns - patches etc - this is not supposed to be some half arsed pipeline; with llamacpp we had a rigorous testing method - have we abandoned here'. Related: MEN26 (promotion of what Flash-Next serves; its step list overlaps steps 1 and 4), MEN27 (release bundle), MEN14 (lab scripts to commands). One build only: PR #637 removes source.radiance-lab. Do step 2 after step 1 or CI goes red.

PROGRESS 2026-11-05: owner chose option (c), carry-forward, and no llama.cpp bump. Done: profile-evidence records carry forward across pin bumps when only the pin differs, rad0008 got its promotion record (PR #642); per-engine evidence gate before materialisation and pin-bump re-audit, config/release.toml evidence-gate required per engine: radiance enforced, llamacpp pending-requalification (PR #649); the v1.17.0 release record carries evidence_gate = passed with rad0002 (qualified at v1.14.0) and rad0008 (v1.9.0) carried forward (bc-radiance-1170.0.0). Remaining: llama.cpp re-qualification (54 of 61 validated patches do not qualify at b11552), its own item, then switch llamacpp to enforced; optional current-pin re-measurement of rad0002/rad0008 (revalidate on request).

## Change Log

- 2026-11-05T07:37:48.187895+00:00 (created-by): Created by agent

SCOPE CORRECTION 2026-11-05 (GPT inventory, req_build-evidence-gate_542c49be5f89, report /work/development/tmp/bigcherry/gpt-bump/REPORT-evidence-gate.md; one patch re-checked by hand with `patch-gates --intent build --source bigcherry`): the gap is not radiance-only. At the current llama.cpp pin b11552, 56 of the 61 validated patches in the production source fail G4 as well (stale base_ref and base_revision, source materialization identity mismatch, stale validation and subject digests, missing HI83 or architecture evidence). llama.cpp's rigorous method exists and was applied at promotion time, but its evidence is not re-qualified when the pin moves, and the build path enforces it for neither engine. Enforcing the gate now would fail every production build of both engines, so it lands only after re-qualification or as a per-engine rollout.

Decision needed from the owner: (a) re-qualify both engines' sets at their current pins before enforcing; (b) enforce for radiance first (2 patches, one pin) and bring llama.cpp up as its own item; or (c) make carry-forward across pin bumps the policy (the bump re-binds evidence by recorded identity checks) and fix why carry-forward does not accept these records. The offline wiring (build gate, pin-bump re-audit) is built after that choice on branch feat/build-evidence-gate, which is empty today.
- 2026-11-05T22:17:58.752020+00:00 (updated-by): Updated: section:notes
