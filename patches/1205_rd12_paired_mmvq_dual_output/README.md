# 1205_rd12_paired_mmvq_dual_output: fuse paired mmvq matmuls over a shared activation (RD12)

Patch id: `1205_rd12_paired_mmvq_dual_output`. Original plan item `RD12` is
superseded by `PRBE11` (docs/planning/active/patching-rdna-boost-experiments/PRBE11.md,
capability-rebaseline-v3-2026-09) -- PRBE11 is the authoritative tracking
item now, not RD12. The `RD12-PAIRED-MMVQ-DUAL` Experiment Contract is
bound in `patch.toml`, and the bespoke correctness producer is
`run_rd12_correctness_check()` in
`tools/bigcherry/patch/validation_campaign.py` (invoked via
`--run-rd12-contract`, or standalone via
`tools/lab/rd12-correctness/run_real.py`). Its `validation.toml` wires
`apply`/`build`/`activation`/`correctness`/`performance`/`controls`.

**What PRBE11 actually requires (per PRBE11.md)**: PRBE11 is the
authoritative active successor to RD12 and is **independent of
PRBE19/RD25** -- PRBE11's own text says it "must not invent an RD25
dependency" (historical RD25 does not touch the paired-MMVQ
implementation) and its acceptance criteria state "RD25/PRBE19 is not a
prerequisite". Its real bar is: exact K/V paired-MMVQ patterns correct
under the declared composition, unsafe/near-miss graphs fall back,
isolated causal performance, and 1207 never silently combined (the
declared `CONFLICTS`). The real correctness and activation evidence below
closes PRBE11's correctness legs on all three contract architectures; the
remaining gap is real causal **performance** evidence (the declared
`performance`/`controls` checks).

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
commit `44b51c66a` (https://github.com/stew675/llama.cpp); not merged into
ggml-org/llama.cpp master. The fork's own claim (not yet independently
verified): bit-identical output vs. the unfused path across its own
1194/1194 MUL_MAT backend tests, plus small positive tg64 gains on gfx1201.

**Conflict**: declared `CONFLICTS = ("1207_rd17_moe_topk_down_fold",)` in
`patch.py` -- RD12's `common.cuh` anchor is inserted against the base
struct shape (no RD17 `x_scale_channel_dst` field), so selecting RD12 and
RD17 together fails the anchor loudly by design; composing them is a
deferred future decision, not an accident to route around.

## How to invoke validation

The contract-bound correctness/activation path is `--run-rd12-contract`
(exactly one contract architecture per run: gfx1100, gfx1201, or gfx1030):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1205_rd12_paired_mmvq_dual_output \
  --run-rd12-contract \
  --hip-path <production-rocm> --amdgpu-targets <gfx1100|gfx1201|gfx1030> \
  --workdir <fresh-workdir>
```

The standalone lab driver used for the 2026-09-13 real runs calls the same
producer for all three architectures in sequence into ONE shared run dir
(writing under `artifacts/lab/rd12-correctness/`) -- the per-arm logs below
are therefore namespaced by architecture so each later run cannot
overwrite an earlier one:

```
PYTHONPATH=tools python tools/lab/rd12-correctness/run_real.py
```

A real `--run-rd12-contract` run binds its evidence into the record: a
canonical `correctness.json` (the bit_identical disposition),
`activation.json`, and one raw per-arm activation log
(`activation-rd12-{subject,control}.log`) bound as the declared
trace-marker check's positive/negative artifacts -- the validator
re-reads those logs and re-verifies the marker itself, so the check
cannot be satisfied by fixture output. The declared
`performance`/`controls` checks stay BLOCKED until real performance
evidence exists.

Activation is proven via the `BIGCHERRY_PATCH_HIT patch=1205_rd12
path=dual_output_mmvq_fusion` trace marker this patch's own
`ggml_cuda_try_fuse` instrumentation emits (once, guarded by an atomic
flag) the first time the dual-output fusion path is actually taken under
`BIGCHERRY_PATCH_TRACE=1` -- this proves the silent, host-side-selected
fusion path was really exercised, not merely that the build succeeded and
a benchmark ran without crashing.

The remaining gap to full PRBE11 qualification is real performance
evidence (the declared `performance`/`controls` checks), tracked under
PRBE11. Nothing prerequisites that work on RD25 (see the PRBE11 note
above) -- it simply has not been measured yet.

## Real bit-identical correctness evidence (2026-09-13, all three architectures)

The correctness-producer piece named above as not-yet-done now exists:
`run_rd12_correctness_check()` (`tools/bigcherry/patch/validation_campaign.py`)
uses a new diagnostic support patch, `1258_rd12_paired_mul_mat_test_case`
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
does NOT by itself satisfy PRBE11's full qualification bar**, which also
requires isolated causal **performance** evidence under the declared
composition. No real performance evidence has been gathered (the
declared `performance`/`controls` checks remain BLOCKED). This closes
the "prove the fork's bit-identical claim on real hardware" gap
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

From this run forward, `--run-rd12-contract` persists one raw per-arm
activation log per architecture
(`logs/activation-rd12-{architecture}-{subject,control}.log`) in the
campaign run directory and binds them into the record's trace evidence,
so the declared trace-marker check can independently re-verify the
marker from the real subprocess output.

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

A/B/C decode comparison on gpt-oss-20B/gfx1100 (RD12's activation was
never confirmed on this model -- round 1's zero hits were the
log-level bug described above, not evidence of non-activation -- so
this measures baseline health, not RD12's own effect): A=178.09,
B=178.01, C=177.48 -- all within noise, no baseline concern for this
patch's domain.
