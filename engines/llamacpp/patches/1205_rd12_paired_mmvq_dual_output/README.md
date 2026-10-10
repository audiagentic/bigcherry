# 1205_rd12_paired_mmvq_dual_output: fuse paired mmvq matmuls over a shared activation (RD12)

Patch id: `1205_rd12_paired_mmvq_dual_output`. Original plan item `RD12` is
superseded by `PRBE11` (docs/planning/active/patching-rdna-boost-experiments/PRBE11.md,
capability-rebaseline-v3-2026-09) -- PRBE11 is the authoritative tracking
item now, not RD12. The `RD12-PAIRED-MMVQ-DUAL` Experiment Contract is
bound in `patch.toml`, and the bespoke correctness producer is the
patch-local `validation/producer.py` (invoked via
`--validation-producer 1205_rd12_paired_mmvq_dual_output/rd12`, or
standalone via `tools/lab/rd12-correctness/run_real.py`). Its `validation.toml` wires
`apply`/`build`/`activation`/`correctness`/`performance`/`controls`.

**What PRBE11 actually requires (per PRBE11.md)**: PRBE11 is the
authoritative active successor to RD12 and is **independent of
PRBE19/RD25** -- PRBE11's own text says it "must not invent an RD25
dependency" (historical RD25 does not touch the paired-MMVQ
implementation) and its acceptance criteria state "RD25/PRBE19 is not a
prerequisite". PRBE11's full bar (its own Steps/Validation/Acceptance
sections) is itemized below -- the real 2026-09-13 evidence covers only
the first bullet, and nothing here claims more:

- **Covered:** one exact-pattern positive case -- the registered 1258
  `test-backend-ops` case (two distinct Q6_K projections over one shared
  F32 activation, the same-shape K/V-projection production shape) is
  bit-identical against the unfused numerical reference (6/6 rows on all
  three contract architectures), with real activation proof (subject
  marker hit, control zero-hit) in the same runs.
- **Not yet covered (non-performance):** the remaining exact-pattern
  gates beyond that single positive case (the full compatible op/type
  and safe view/data-interval gate set, same-source
  overlap/disjoint-output gates); unsafe/near-miss graph fallback to the
  unfused reference (false-positive handling, GLU fusion precedence,
  views/no-ops); graph capture as an explicit validation leg; and the
  declared-composition arms (isolated vs. declared-composition causal
  arms, per PRBE11's step 5).
- **Not yet covered (performance):** real causal performance evidence
  (the declared `performance`/`controls` checks) -- isolated performance
  and any composition result separately attributable.
- **Met structurally:** 1207 is never silently combined -- the declared
  `CONFLICTS = ("1207_rd17_moe_topk_down_fold",)` fails the anchor
  loudly, and no dedicated recipe declares an order combining the two.

## Scope

`patch.toml` declares `validation-architectures = ["gfx1100", "gfx1201",
"gfx1030"]`. The fork's own reported performance evidence (see below) is
specific to gfx1201 and has not been independently measured on this
project's own hardware; the real correctness/activation evidence below was
produced on all three declared architectures. Backend: HIP.

## What it does

When two adjacent `MUL_MAT` nodes share the same activation and output
shape and are mmvq-eligible (e.g. the K and V projections of an attention
layer), the first matmul also computes the second weight's result as a
fusion "gate" and writes it to a separate `dst_gate` destination instead of
combining it into the main output -- one launch instead of two, avoiding a
second read of the shared activation. Ported from stew675-rdna-boosts fork
commit `44b51c66a` (<https://github.com/stew675/llama.cpp>); not merged into
ggml-org/llama.cpp master. The fork's own claim (not yet independently
verified): bit-identical output vs. the unfused path across its own
1194/1194 MUL_MAT backend tests, plus small positive tg64 gains on gfx1201.

**Conflict**: declared `CONFLICTS = ("1207_rd17_moe_topk_down_fold",)` in
`patch.py` -- RD12's `common.cuh` anchor is inserted against the base
struct shape (no RD17 `x_scale_channel_dst` field), so selecting RD12 and
RD17 together fails the anchor loudly by design; composing them is a
deferred future decision, not an accident to route around.

## How to invoke validation

The contract-bound correctness/activation path is the patch-local
producer (exactly one contract architecture per run: gfx1100, gfx1201,
or gfx1030, named by `--amdgpu-targets` and mapped to a real device by
`--device-map`):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1205_rd12_paired_mmvq_dual_output \
  --validation-producer 1205_rd12_paired_mmvq_dual_output/rd12 \
  --device-map <gfx1100|gfx1201|gfx1030>=<device-index> \
  --hip-path <production-rocm> --amdgpu-targets <gfx1100|gfx1201|gfx1030> \
  --workdir <fresh-workdir>
```

The standalone lab driver calls the same producer once per contract
architecture in sequence (writing under `artifacts/lab/rd12-correctness/`),
sharing the worktree/build roots so the fat multi-arch binaries are built
once and reused while each run dir stays per-architecture (the producer
namespaces its own artifacts by architecture either way):

```
PYTHONPATH=tools python tools/lab/rd12-correctness/run_real.py
```

A real `--validation-producer 1205_rd12_paired_mmvq_dual_output/rd12`
run binds its evidence into the record through the shared binder: a
canonical `correctness.json` (the bit_identical disposition),
`activation.json`, and one raw per-arm activation log
(`artifacts/activation-rd12-{architecture}-{subject,control}.log`) bound
as the declared trace-marker check's positive/negative artifacts -- the
validator re-reads those logs and re-verifies the marker itself, so the
check cannot be satisfied by fixture output. The declared
`performance`/`controls` checks stay BLOCKED until real performance
evidence exists.

Activation is proven via the `BIGCHERRY_PATCH_HIT patch=1205_rd12
path=dual_output_mmvq_fusion` trace marker this patch's own
`ggml_cuda_try_fuse` instrumentation emits (once, guarded by an atomic
flag) the first time the dual-output fusion path is actually taken under
`BIGCHERRY_PATCH_TRACE=1` -- this proves the silent, host-side-selected
fusion path was really exercised, not merely that the build succeeded and
a benchmark ran without crashing.

The remaining gaps to full PRBE11 qualification are itemized above --
the non-performance exact-pattern/fallback/composition gates plus the
real causal performance evidence (the declared `performance`/`controls`
checks) -- all tracked under PRBE11. Nothing prerequisites that work on
RD25 (see the PRBE11 note above) -- it simply has not been covered yet.

## Real bit-identical correctness evidence (2026-09-13, all three architectures)

The correctness-producer piece named above as not-yet-done now exists:
the patch-local producer `validation/producer.py` (PA36 sub-slice 2;
previously `run_rd12_correctness_check()` in shared code, deleted with
the `--run-rd12-contract` CLI path) uses a new diagnostic support patch,
`1258_rd12_paired_mul_mat_test_case`
(a registered whole-graph `test-backend-ops` case with two distinct Q6_K
MUL_MAT projections over one shared F32 activation, matching the real
K/V-projection production shape), to prove the fork's bit-identical claim
via exact digest equality plus a real activation-marker check (the focal
patch's own `BIGCHERRY_PATCH_HIT` trace, gating a false-green result if
the fusion never actually activated).

Real runs on Brutus across all three architectures: **clean PASS on
gfx1100, gfx1201, and gfx1030** -- 6/6 (projection, seed) rows
byte-identical (both K and V lanes, 3 seeds each) on every architecture,
with real activation confirmed (subject hit, control miss) in every run.
One real bug was found and fixed along the way: the test case's first
version joined K/V with a terminal ADD, which made the second MUL_MAT
eligible for the pre-existing CUDA MUL_MAT+ADD fusion and left `v_out`
unmaterialized (a ~800 NMSE on the CONTROL build, not a RD12 defect) --
fixed by making both outputs independent graph roots (see 1258's own
README for the full finding).

**This is real, clean multi-architecture correctness evidence -- but it
does NOT by itself satisfy PRBE11's full qualification bar**, which
also requires the remaining exact-pattern/fallback gates and isolated
causal **performance** evidence under the declared composition (see the
itemized list above). No real performance evidence has been gathered
(the declared `performance`/`controls` checks remain BLOCKED). This
closes the "prove the fork's bit-identical claim on real hardware" gap
specifically -- it does not by itself authorize promotion.

## Known limitations

Not `deferred-hardware`. The fork's own performance claim (small positive
tg64 gains on gfx1201) has not been independently measured on this
project's hardware -- the real evidence above covers correctness and
activation only, not performance.

## Evidence

**Real apply/build/activation check (2026-09-13, current pin b10901/28ff0958291c).**
Built real control (baseline) and subject (1205 applied) `llama-bench`
binaries on Brutus. Apply and build both real, clean PASS (both
`bigcherry-native` compositions configured and compiled cleanly).

Activation, round 1 (2026-09-13, real, negative -- METHODOLOGY BUG, not
a real negative): ran `BIGCHERRY_PATCH_TRACE=1 llama-bench -p 512 -n 32`
(no `--verbose`) against two real models (`Qwen3.5-4B-UD-Q6_K_XL` and
`gpt-oss-20b-UD-Q6_K_XL`, single gfx1100): **0 of 0**
`BIGCHERRY_PATCH_HIT patch=1205_rd12` marker hits in either run.

**Root cause found and corrected: at the time, this patch's marker used
`GGML_LOG_INFO`, not `GGML_LOG_WARN`** -- the exact same class of bug
this project already found and fixed for RD08
(`GGML_LOG_INFO`->`GGML_LOG_WARN`, commit `991e761`, VA21): `llama-bench`
gates INFO-level ggml logs behind its own `--verbose` flag, which the
round-1 command above did not pass. **This was a real methodology bug in
my own test command, not a real negative finding about the patch or the
model.** (The marker now uses `GGML_LOG_WARN` -- see the round-2 entry
below.)

**Activation, round 2 (2026-09-13, real, POSITIVE, all three
architectures).** Rerunning the identical command with `--verbose` added,
against the SAME `Qwen3.5-4B-UD-Q6_K_XL` model already registered as
`tierA-qwen4b-q6k` (confirmed elsewhere this session, via PRBE102, to be
a dense+GDN hybrid, not purely dense): **clean, real positive/negative
split on all three available architectures** --
`BIGCHERRY_PATCH_HIT patch=1205_rd12 path=dual_output_mmvq_fusion` fired
once (subject_hit=1) on gfx1100, gfx1201, AND gfx1030, zero hits
(control_hit=0) on the baseline build on all three. **RD12's activation
leg is resolved: the patch's fusion genuinely activates, on real
hardware, across every available architecture.** The `GGML_LOG_INFO` ->
`GGML_LOG_WARN` fix this trap pointed at was made the same day (PRBE106,
2026-09-13; the marker now uses `GGML_LOG_WARN`, matching RD08's
precedent).

No performance qualification has been attempted yet -- the declared
`performance`/`controls` checks remain BLOCKED, and PRBE11 does not
prerequisite that work on RD25 (see above), so it is open work rather
than a blocked prerequisite. The 2026-09-13 correctness runs above were
done as real bit-identical proof of the fork's claim, not as a promotion
decision.

From this run forward, the producer path persists one raw per-arm
activation log per architecture
(`artifacts/activation-rd12-{architecture}-{subject,control}.log`) in
the campaign run directory and binds them into the record's trace
evidence, so the declared trace-marker check can independently
re-verify the marker from the real subprocess output.

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

A/B/C decode comparison on gpt-oss-20B/gfx1100 (RD12's activation was
never confirmed on this model -- round 1's zero hits were the
log-level bug described above, not evidence of non-activation -- so
this measures baseline health, not RD12's own effect): A=178.09,
B=178.01, C=177.48 -- all within noise, no baseline concern for this
patch's domain.

## Migrated generic-producer path: hardware equivalence receipt (2026-09-17, gfx1030)

The bespoke `--run-rd12-contract` path was deleted and this patch's
correctness now runs through the generic validation-producer framework
(PA36 sub-slice 2, commit a162da91; see "How to invoke validation"
above). The real-hardware equivalence re-run required by PA36's
migration design (dev-gpt-agent req_4c04142630ad45ce: gfx1030
sufficient to close -- the new path's architecture-specific behavior is
limited to validating/selecting the one execution architecture, while
the correctness build remains the same fat
{gfx1100,gfx1201,gfx1030} pair) ran on Brutus gfx1030 (device 3,
6900 XT) against current pin b10901/28ff0958291c: five-build standard
scaffold, the fat-three `test-backend-ops` correctness pair, 6/6
(projection, seed) rows **bit-identical** with real activation
(subject 6/6 marker hit, control zero-hit), record persisted, exit 0
(ineligible as designed -- performance/controls still unsatisfied,
contract verdict blocked/no-promotion, exactly as the 2026-09-13
evidence).

Record-vs-record (new record vs the 2026-09-13 accepted gfx1030 record,
same `evidence/validation.json` log): correctness disposition
`passed`/mechanism `rd12-paired-mmvq-bit-identical`, activation
`activation-verified` with byte-identical detail, role sets
{tune,replay,stock}/{control,subject}, apply/build/correctness pass,
performance/controls unsatisfied, base_ref/base_revision -- all
semantically identical. `record_digest` differs (required by the new
validation implementation); `campaign_identity_digest` differs
(permitted -- the model-free schema replaced the RD12-specific one);
the scaffold baseline source is `bigcherry-tuning` (the PA29 cutover
moved 0110's campaign plumbing out of plain `bigcherry` between the
two runs -- the correctness PAIR itself is unaffected, built by the
producer with its own `bigcherry` baseline). Two evidence-strength
additions in the new path (GPT-confirmed as additions, not new
measurement criteria): the declared activation trace-marker check is
now actually evaluated and PASSes by re-reading the bound raw logs
(the 2026-09-13 record left that check `blocked` while recording the
same activation evidence), and the two raw per-arm activation logs are
bound as declared producer artifacts, exposing the already-existing
underlying marker evidence through the declarative artifact contract.

One real bug was found by this hardware run and fixed before the
equivalence result (commit 240153e9): `StandardCampaignScaffold` had
lost its `@dataclass` decorator in a162da91, so the first hardware run
died after all five builds with "takes no arguments"; restored plus a
structural regression test.

Coverage note: the gfx1100 and gfx1201 legs are the 2026-09-13
accepted historical coverage (not freshly re-validated through the
migrated path) -- a fresh generic-path gfx1100 run is optional
strengthening only (at the time of this run the Brutus dual-7900 XTX
was occupied; the local Windows host with ROCm 7.1 is available for it).
gfx1201 was excluded from fresh runs by session constraint.
