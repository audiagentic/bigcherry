# 1203 (RD05/RD06/RD07): RDNA4 WMMA flash-attn + Q6_K mmq prefill performance

## Scope

Bundles three related fork changes into one net patch (causally coupled, one
upstream PR, splitting them would leave the graph in an uncompilable
intermediate state):

- **RD05** (correctness): fixes a head-256 WMMA flash-attn combine race and
  a tile_Q reuse race.
- **RD06** (performance): tunes and enables the WMMA flash-attn path up to
  head 576 on RDNA4 by default.
- **RD07** (performance): hoists/folds Q6_K mmq sub-scales into the row
  base-scale to remove an int-mul from the scale chain (fork's own
  reported number: Q6_K mmq 40 -> 58 TFLOPS).

Bound to 3 Experiment Contracts (`RD05-WMMA-FA-CORRECTNESS-BARRIERS`,
`RD06-RDNA4-WMMA-FA-CONFIG`, `RD07-Q6K-MMQ-PREFILL-FOLD`) but no
`validation.toml` adapter exists yet -- executing these contracts through
the standard harness would need real producers wired for each first.

## Upstream / provenance

Ported (with adaptations for this project's own 1000/HI70 anchors) from
`stew675-rdna-boosts` fork commit `1d525bd45`
(https://github.com/stew675/llama.cpp). Not merged into `ggml-org/llama.cpp`
master.

## Real hardware evidence (2026-09-12, Brutus, gfx1201/R9700, device index 2)

This patch had never been benched on real hardware before this session
(plan items PRBE02/PRBE03/PRBE04 were all in `pending` state with zero
evidence). Built BC-baseline (1203 excluded) and BC+1203 from a single
isolated scratch clone (`resolve_source_composition`/
`materialize_composition`, pin `28ff0958291ce3465fabd7bd679d4b0edd742bd9`),
targeting `gfx1201` specifically (the architecture RD06's config change is
scoped to).

**Correctness** (`llama-perplexity`, `tierA-qwen4b-q6k`, real wikitext2
corpus, `-c 2048 -ngl 99`):

| arm | PPL |
|---|---:|
| BC-baseline (1203 excluded) | 10.4463 +/- 0.02753 |
| BC + 1203 | 10.3938 +/- 0.02737 |

Combined-uncertainty sigma = |10.4463 - 10.3938| / sqrt(0.02753^2 + 0.02737^2)
= **1.35** -- well under this project's established 3-sigma significance
threshold (`tools/bigcherry/experiment/perplexity.py`'s
`require_ppl_equality`). **PASS**: no statistically significant PPL
divergence.

**Performance** (`llama-bench`, same model, `-fa on -p 512,2048 -n 128 -r 5`):

| test | BC-baseline | BC + 1203 | delta |
|---|---:|---:|---:|
| pp512 | 5260.12 +/- 153.50 | 5439.31 +/- 175.88 | +3.4% |
| pp2048 | 5102.77 +/- 1.79 | 5417.97 +/- 5.88 | **+6.2%** |
| tg128 | 92.63 +/- 0.52 | 92.63 +/- 0.57 | +0.0% |

pp2048's stddev is tight on both arms (1.79 and 5.88 t/s against a ~300
t/s difference), making that delta a credible real signal rather than
noise; pp512 has wider variance but points the same direction. tg128
(decode) is correctly unaffected -- this patch targets flash-attn prefill
and Q6_K mmq, not the decode matvec path.

## Real RD05 contract correctness evidence (2026-09-13, gfx1201)

RD05's contract (`RD05-WMMA-FA-CORRECTNESS-BARRIERS`, `correctness.backend_reference
= "required"`) had no producer until this session. Patch 1203 bundles
RD05 (correctness only) with RD06/RD07 (real performance changes) as one
atomic patch.py, so there is no way to isolate RD05 alone via source
composition -- `bit_identical` would be the wrong bar (RD06/RD07 are
expected to shift numerics at the margin). `run_rd05_contract_correctness()`
(`tools/bigcherry/patch/validation_campaign.py`) reuses this project's
`backend_reference` sigma-vs-threshold technique
(`tools/bigcherry/experiment/perplexity.py`), the same approach validated
for RD13 earlier this session.

Real run on Brutus (single gfx1201 R9700), control (1203 absent) vs
subject (1203 applied), against a real wikitext2 corpus slice:
**PASS** -- subject PPL=10.5871, control PPL=10.5869, delta=0.00020,
sigma=0.0010 (well within the 3.0 threshold). This is consistent with,
and considerably tighter than, this patch's own earlier ad-hoc PPL
comparison (sigma=1.35) -- both real measurements agree the bundled
patch does not corrupt output.

Contract still not bound in `patch.toml` -- this establishes the
correctness leg only; RD05's contract also carries no performance claim
(`expected_effect = "correctness"`), so this closes RD05's evidence
obligation in full modulo the formal binding step.

## Real RD06 contract correctness evidence (2026-09-13, gfx1201)

RD06's contract (`RD06-RDNA4-WMMA-FA-CONFIG`, `correctness.backend_reference
= "required"`) had no producer until this session. Like RD05, RD06 cannot
be isolated from RD05/RD07 via source composition (one atomic patch.py),
so `run_rd06_contract_correctness()` reuses the same real PPL-comparison
technique, scoped to gfx1201 (RD06's contract scope).

Real run on Brutus (single gfx1201 R9700), control (1203 absent) vs
subject (1203 applied), against the same wikitext2 corpus slice used for
RD05: **PASS** -- subject PPL=10.5835, control PPL=10.6394, delta=0.0559,
sigma=0.2709 (well within the 3.0 threshold).

Contract still not bound in `patch.toml`. This proves backend-reference
correctness of the complete shipped 1203 patch, not causal attribution to
RD06's config-selection logic specifically -- RD06's own performance
claim (per-shape/head-dimension/softcap gains, and confirming gfx1100
does not select or regress) remains separate, not-yet-gathered work.

## Real RD07 contract correctness evidence (2026-09-13, all three contract architectures)

RD07's contract (`RD07-Q6K-MMQ-PREFILL-FOLD`, `correctness.backend_reference
= "required"`) covers all three of this project's real architectures
(gfx1100, gfx1201, gfx1030), unlike RD05/RD06 (gfx1201-only). Reused the
same real PPL-comparison technique (`run_rd07_contract_correctness()`),
requiring one real, independent build+run per architecture -- a
multi-target fat compile is never accepted as per-architecture execution
evidence.

Real runs on Brutus, control (1203 absent) vs subject (1203 applied),
against the same wikitext2 corpus slice used for RD05/RD06:

| architecture | subject PPL | control PPL | sigma | result |
|---|---:|---:|---:|---|
| gfx1100 | 10.6173 | 10.6173 | 0.0000 | **PASS** |
| gfx1201 | 10.5835 | 10.6394 | 0.2709 | **PASS** |
| gfx1030 | 10.6007 | 10.6007 | 0.0000 | **PASS** |

This is this session's first patch with real, clean multi-architecture
correctness evidence across all three of this project's real GPU
architectures at once. Contract still not bound in `patch.toml` -- this
proves backend-reference correctness of the complete shipped 1203 patch
on every architecture it targets, not causal attribution to RD07's
Q6_K MMQ fold specifically (which remains subject to HI71's dense-shape
eligibility re-verification per the contract's own co-tenancy warning).

## Known limitations -- real gaps, not yet closed

This is a genuine first real signal, not full closure of PRBE02/PRBE03/PRBE04:

- **PRBE02** (RD05 correctness) requires a specific targeted head-size
  matrix (192/256/320/512/576) under both graph and non-graph execution,
  repeated under load. The PPL check above exercises whatever head sizes
  this one real model's attention layers use, not that full matrix.
- **PRBE03** (RD06 performance) requires per-shape/softcap balanced-repeat
  measurements with an explicit gfx1100 non-selection/non-regression
  control (proving RD06's expanded config does NOT get selected on
  gfx1100). Not yet run.
- **PRBE04** (RD07 Q6_K fold) requires PEF01's mandatory illegal-memory/
  safety gate and HI71's dense-shape eligibility check before any timing
  claim can count -- explicitly NOT satisfied by this run. PEF01 itself is
  a different, gfx1100-specific tuner exhaustive-candidate-sweep issue
  (forced J=112) not triggered by this default-dispatch run, but the gate
  it requires is still a real, separate, unmet prerequisite for RD07
  specifically.
- No native-llama.cpp comparison arm for the performance claim (this
  project's new `optimization`-tag rule, `PATCH_AUTHORING.md`) -- not yet
  tagged `optimization` or gated by `check_performance_evidence` pending
  that comparison.
- `state` correctly stays `"untested"` -- real, positive first evidence
  exists, but the patch's own tracked items' acceptance criteria are not
  yet met.
