# 1205_rd12_paired_mmvq_dual_output: fuse paired mmvq matmuls over a shared activation (RD12)

Patch id: `1205_rd12_paired_mmvq_dual_output`. Original plan item `RD12` is
superseded by `PRBE11` (docs/planning/active/patching-rdna-boost-experiments/PRBE11.md,
capability-rebaseline-v3-2026-09) -- PRBE11 is the authoritative tracking
item now, not RD12. No Experiment Contract is bound yet -- this patch has
no `experiment-contract` field in `patch.toml` and no bespoke correctness
producer under `validation/`, unlike RD04/RD08/RD58/RD73. Its
`validation.toml` wires only `apply`/`build`/`activation` checks.

**HARD PREREQUISITE, not yet met (PRBE11's own steps)**: RD25's batch-vs-seq
consistency fix (fork commit `8cdf1ab08`) must land first -- the frozen
paired-mmvq code this patch ports is known to carry that inconsistency, and
PRBE11 explicitly requires it be "resolved" before qualification. RD25 has
not been ported as a standalone patch yet (it is currently only a bake-in
rule referenced in this patch's own provenance notes). **Do not run a real
qualification campaign for a promotion decision on this patch until RD25 is
resolved** -- the generic S1-S7 campaign below (apply/build/activation) is
still safe to run for basic package-health/activation confirmation, but its
result cannot stand in for the real correctness/performance qualification
PRBE11 actually requires.

## Scope

No `validation-architectures` are declared in `patch.toml` yet. The fork's
own reported evidence (see below) is specific to gfx1201; that has not been
independently confirmed on this project's own hardware. Backend: HIP.

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

No contract-bound qualification path exists yet -- only the generic
S1-S7 campaign (apply/build/activation) is currently invocable:

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1205_rd12_paired_mmvq_dual_output \
  --model <a real .gguf> \
  --hip-path <production-rocm> --amdgpu-targets <target> \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --worktree-root <worktree-root>
```

Activation is proven via the `BIGCHERRY_PATCH_HIT patch=1205_rd12
path=dual_output_mmvq_fusion` trace marker this patch's own
`ggml_cuda_try_fuse` instrumentation emits (once, guarded by an atomic
flag) the first time the dual-output fusion path is actually taken under
`BIGCHERRY_PATCH_TRACE=1` -- this proves the silent, host-side-selected
fusion path was really exercised, not merely that the build succeeded and
a benchmark ran without crashing.

Bringing this patch to RD08's level of qualification (a real bit-identical
correctness producer proving the fork's own claim, a bound Experiment
Contract, and real performance evidence) is separate, not-yet-done
authoring work tracked under PRBE11 -- blocked on the RD25 prerequisite
above before that work is meaningful.

## Known limitations

Not `deferred-hardware`. No real fresh evidence exists yet for this
project's own hardware -- everything under "What it does" attributed to
"the fork" is the fork's own reported claim, not this project's
independently-measured result.

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

**Root cause found and corrected: this patch's marker uses
`GGML_LOG_INFO`, not `GGML_LOG_WARN`** (`patch.py` line 155) -- the exact
same class of bug this project already found and fixed for RD08
(`GGML_LOG_INFO`->`GGML_LOG_WARN`, commit `991e761`, VA21): `llama-bench`
gates INFO-level ggml logs behind its own `--verbose` flag, which the
round-1 command above did not pass. **This was a real methodology bug in
my own test command, not a real negative finding about the patch or the
model.**

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
hardware, across every available architecture.** The `mmvq.cu` marker
itself should be changed from `GGML_LOG_INFO` to `GGML_LOG_WARN` to match
RD08's precedent and avoid this exact trap recurring for future
validators (not yet done -- a real, small, low-risk authoring fix,
tracked below).

Per this patch's hard prerequisite (RD25 not yet ported), no
correctness/performance qualification was attempted -- that remains
deliberately scoped out until RD25 lands.

Runtime artifacts (raw logs) recorded but not yet persisted as a durable
evidence bundle; further root-causing the zero-activation result is real,
not-yet-done work (a candidate task for whoever picks up PRBE11's RD25
prerequisite, since activation must be confirmed before any correctness
work on this patch is meaningful).

## Real three-arm baseline comparison (2026-09-13, standardized criteria)

A/B/C decode comparison on gpt-oss-20B/gfx1100 (RD12's fusion does not
activate on this model -- see above -- so this measures baseline health,
not RD12's own effect): A=178.09, B=178.01, C=177.48 -- all within
noise, no baseline concern for this patch's domain.
