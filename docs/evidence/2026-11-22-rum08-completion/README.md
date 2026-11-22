# RUM08: fixed-depth MTP replay on Radiance 2.2 / bc-radiance 2020.0.0

Evidence date: 2026-11-22. Source: upstream Radiance 2.2.0 at pinned
`b6b8751cccb0e834a13ae9e5afca824ee9a487aa`; installed production
bundle: `bc-radiance-2020.0.0` (ROCm 7.2.4). The release numbering
now uses the four-digit `2020.x` line, not the older 1170.x line.

## Disposition and production policy

- **rad0013** is archived unchanged and superseded by the upstream
  `qwen4exp_fp8::pass_key` + `Ctx::pass_key`/recorded-tape implementation.
  It is absent from the shipped recipe. The old 1.17 gain is historical and
  is not an additional replay mechanism in 2.2.
- **rad0047** remains archived unchanged and rejected. Valid wide
  scheduler-produced verify entries carry a nonempty full-KV host summary;
  a direct synthetic null/zero callback input is not such a scheduler entry.
- **rad0036** stays composed, not superseded by upstream timing switches.
  The production `flashnext-4gpu` model selects **fixed speculative depth 3,
  prefill-interleave 0, mixed-step-target-ms 0, and default arrival
  admission**. The optional governor's budget/age/remaining policy is *not*
  selected in this profile. In the separate Qwen27B long profiles, rad0036
  controls admission and upstream interleave controls step spacing. These
  are distinct controls, not competing admission authorities.
- No adaptive MTP depth, suffix-specific or traversal policy was enabled
  in the fixed-depth Flash-Next reproduction. No new production feature
  or hot-path dispatch was added for this evidence.

## Card-free actual-source checks

- `tools/tests/patch/test_radiance_upstream_mtp_replay.py`: **10 passed**,
  compiling the actual pinned qwen4exp `pass_key` and the **real**
  `abi/rad_prefill_split.h` split and grid functions, not a fabricated
  splitter. It validates ordinary decode, wide speculative verify,
  history (-1), fixed serial MTP draft passes (2, 3), QSA exact-bound
  selector crossing, refusal of unsafe mixed/pre-fill speculative replay,
  prefill equal/different split grids, scheduler metadata and patch
  composition.
- Earlier host baseline: **20 passed** across this callback fixture,
  rad0028 rank-consistent drafts, rad0029 rank ownership and rad0036
  governor tests. The strengthened actual split test is green separately.
- Compiled **unmodified upstream runtime test** executable from the
  Radiance 2.2 source: `issue_passes_only_proven_kv_host_summaries`,
  `an_arch_keyed_program_issues_decode_passes_at_the_keyed_bound`,
  `the_prefill_geometry_key_reuses_equal_grids_and_separates_changed_grids`:
  **3 cases, 30 checks passed**. Unmodified upstream `arch_test` QSA
  selector case: **1 case, 30 checks passed**. The actual runtime test
  checks `RadArgs.kv_used_max_h`: positive decode, verify, draft launches
  get the keyed `max_ctx_len + max_q_len` (test case 2050), not moving
  live-KV 1152; a prefill/mixed launch keeps the exact host summary.
  The runtime test source hash matched the pinned upstream checkout;
  the architecture test binary came from a patched build and its source
  may have additional unrelated patches, so it is supporting evidence,
  not a separate source-identity claim.
- Upstream `sched_test`: **10 selected cases, 356 checks passed**,
  covering mixed prefill/decode, speculative verification, interleave-off,
  prefix adoption, preemption priorities, preempted prefix publishing
  and resumed output, and budget/waiter interaction. These are host
  scheduler tests, not GPU proof of a mixed scheduler step.
- The actual `Ctx::run_pass` records decode keys on first sight,
  replays a matching tape, periodically audits it against a live
  recording, and faults if the key misses a launch dependency. The
  operator's HIP phase trace itself does **not** log pass keys,
  replay counters or effective `RadArgs`; no trace-only claim is made
  about equality of launch arguments.

## Installed-release hardware

See `hardware-summary.json` for bounded receipts derived from installed
`bc-serve serve CONFIG --model-root MODEL_ROOT --work-root WORK_ROOT
--step-phase-trace` and `rum09_phase_timeline.py`. Each run used its own
short GPU lease and verified owner health, real inference responses and
clean stop. Uninstrumented speedup is **not** claimed: the tracing mode
synchronizes at each pass and changes timing.

The first `flashnext-4gpu` run was healthy and stopped with code 0.
Two deterministic 56-token requests both returned successfully
(624.15 ms cold and 620.47 ms warm). There were 18 decode steps
in each selected trace window and repeated pass shapes -1, 0, 2 and 3,
with the same keyed 1024-token context bound observed on the two
rounds. Recorded-tape log entries occurred for the history -1,
serial draft 2 and 3, and verify 0 passes; no replay-argument mismatch
was logged. The timeline analyzer reported median traced compute
26.959 ms (cold) and 26.886 ms (warm). The phase trace itself does not count tape playback; the released
engine's `/stats` does. On independently started **helper-inactive**
`flashnext-r9700`, two fixed-depth requests returned identical output.
After those requests, the live counters were **171 played, 6 recorded,
6 audited**, and **zero refused or unkeyed**. The long-prefill and
concurrent short request then both succeeded and their HTTP lifetimes
overlapped by 111,021 ms. Final counters were **486 played, 40 recorded,
32 audited, zero refused and zero unkeyed**. The trace had prefill and
decode phases, but no mixed-phase pass. Concurrent client lifetimes
alone do not prove one mixed scheduler step.

The active-helper `flashnext-tp2xtx-r9700` TP2 trace returned
**identical per-pass shapes across ranks 0 and 1**: 19 selected decode
steps per cold/warm round and the same 1024-token context bucket,
history, draft and verify sequence. Both 56-token outputs were
deterministic within each run. Median traced step spans (cold/warm)
were 33.673/33.673 ms on rank 0 and 33.663/33.642 ms on rank 1.
These are instrumented observations, not speedup claims.

The final `flashnext-4gpu` **active-helper** repeat used the same
installed 2020.0.0 release, fixed-depth YAML, and trace flag with the
reverse arrival order (short decode admitted first, long prefill second).
Both repeated 56-token responses and both overlapping responses
succeeded, with a clean stop. After repeat rounds the engine's native
`/stats` counted **175 played, 6 recorded, 5 audited, 0 refused and
0 unkeyed**. The concurrent request windows overlapped by 3.049 s;
the actual GPU trace contained **six `RAD_PHASE_MIXED` passes**.
At completion `/stats` counted **411 played, 22 recorded, 8 audited,
7 unkeyed, and 0 refused**. Those seven live, unkeyed issues are
consistent with the upstream mixed-pass replay guard: a mixed
decode/prefill geometry is ineligible for tape reuse and must issue
live; they are not replay-key faults. The dry single-controller
configuration was unchanged. The mixed-geometry case is not claimed
to have a replay-eligible key.

## Acceptance decision

| Acceptance clause | Verifiable evidence | Status |
| --- | --- | --- |
| rad0013 and rad0047 retirement without losing exactness | Current recipe has neither patch; actual-source callback/real split fixture 10/10; exact-QSA selector, zero/unknown summary and split-grid cases | Met |
| Same effective launch arguments on admitted decode/verify/MTP draft replay | Upstream `runtime_test` tests actual `RadArgs.kv_used_max_h` (3 cases, 30 checks), `arch_test` exact QSA selector (30 checks); recorded tape and positive native `played` counts on installed release | Met for fixed-depth admitted shapes |
| Mixed/nonpositive or unrecordable shape fails closed to live issue | Actual pass-key returns no replay key for mixed/wide speculative prefill; six mixed passes observed on active-helper hardware, seven `issued_unkeyed` with no `issued_refused`; scheduler mixed/preemption tests 356 checks | Met |
| Rank-0 proposal/accepted history and per-rank resource bounds | rad0028/rad0029 host fixture tests passed; TP2 helper-active replay produced *byte-for-byte identical* per-pass shape tuples `(step, draft_pass, phase, n_tok, ctx)` for both ranks, 19 decode steps per round and no error; consistent generated completions | Met |
| Helper on/off and real canary | TP1 helper-active, TP2 helper-active and TP1 helper-inactive; all completed two fixed-depth requests plus long-prefill, and stopped cleanly with exit zero; final helper-active mixed case also succeeded | Met |
| One explicit production scheduler/interleave authority | `flashnext-4gpu` YAML explicitly fixes depth 3, interleave 0 and mixed target 0; arrival admission; rad0036 optional long policy dormant here; no adaptive/suffix/traversal controller; policy documented in Radiance reference | Met |
| Honest timing and replay provenance | `phase-timelines.json` from shipped `rum09_phase_timeline.py` with per-rank compute/collective/wait, and native `/stats` playback counters; HIP sync alters timing; historical Gate D baseline kept separate | Met |

**Technically ready for the lead to complete RUM08** for this scoped
fixed-depth Flash-Next production configuration. The proof does not
qualify future adaptive depth, traversal, suffix proposals or
long-Qwen policies. It also does not establish an uninstrumented
performance gain, nor claim direct kernel argument equality from the
HIP trace alone: actual effective-argument correctness is from the
upstream executable tests and key-bound invariants, supported by
replay counters and healthy hardware canaries.

The lead owns the Planning MCP state transition and Ledger MCP event.
The Planning and Ledger interfaces were not exposed to this task
session, so the authoritative plan state remains unchanged here.
