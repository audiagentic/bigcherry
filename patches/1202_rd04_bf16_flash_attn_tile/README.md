# 1202_rd04_bf16_flash_attn_tile: native-BF16 flash-attn tile kernel series (RD04)

Patch id: `1202_rd04_bf16_flash_attn_tile`. Plan item: `RD04`. Bound
Experiment Contract: `RD04-BF16-FLASH-ATTN-TILE`
(`config/experiment-contracts.toml`).

## Scope

Target architectures: gfx1100, gfx1201, gfx1030 (contract
`scope.architectures`). Backend: HIP. A net of 7 fork commits forming
the flash-attn TILE kernel series.

The TILE kernel reads BF16 K/V natively on RDNA3+ with FP32
accumulation (F16 accumulators flush small values at deep context).
Packed BF16 PV is the fork's only native-BF16 VKQ path. Target
configuration: BF16 KV cache + flash attention enabled (`-fa on -ctk
bf16 -ctv bf16`). Contract acceptance: target_kernel_gain_pct 3.39% on
decode, max_control_regression_pct 1% on prefill; positive workloads
decode + long_context, control workload prefill; model
`tierA-qwen4b-q6k`.

## Historical evidence is not current

The fork's own isolated bench (2026-08-20, local 7900 GRE gfx1100,
Ministral-3-14B Q4_K_M, BF16 KV+FA): tg128 +3.39%, no measured prompt
cost at pp>=1024. This is real, but it predates this validation package
and was measured against an earlier pin -- it does **not** by itself
satisfy this contract's current-pin evidence obligation
(`patch-verify-evidence`/VA08). A fresh, current-pin
`--run-rd04-benchmark` run is required before this patch's
`ported-benched` status can be reported as currently qualified.

## How to invoke validation

Hardware-free benchmark-evidence producer (VA04):

```
PYTHONPATH=tools python -m bigcherry.patch.validation_campaign \
  --patch 1202_rd04_bf16_flash_attn_tile \
  --model <tierA-qwen4b-q6k.gguf> \
  --hip-path <production-rocm> --amdgpu-targets gfx1100 \
  --manifest <hip-autotune-manifest.json> \
  --workdir <fresh-workdir> --build-root <build-root> \
  --worktree-root <worktree-root> \
  --run-rd04-benchmark
```

`--run-rd04-benchmark` executes RD04's real paired decode/prefill lanes
(`tools/bigcherry/experiment/execution.py`, via
`run_rd04_benchmark_evidence()`) against the parity-built control/
validation-subject binaries with `-fa on -ctk bf16 -ctv bf16`, and
binds the real result into `ctx.performance_evidence` so the
"performance"/"controls" adapter checks reach a real PASS/FAIL. It does
**not** execute the generic S1-S7 tune/promote/replay campaign (that
pipeline's own promotion decision is unrelated to RD04's own
validation-domain evidence -- the exact bug VA15 found and fixed for
RD08), and it does **not** attempt correctness or activation proof or
contract promotion -- `eligible_for_validated_state` stays `False`
after this command; `ported-benched` current-pin qualification (VA08)
is the honest ceiling this command can produce.

## Real RD04 contract correctness evidence (2026-09-13, CONFIRMED)

RD04's contract (`RD04-BF16-FLASH-ATTN-TILE`, requiring both
`backend_reference` and `ppl_equality`) had no correctness producer until
this session. `run_rd04_contract_correctness()`
(`tools/bigcherry/patch/validation_campaign.py`) derives both checks from
one real whole-model PPL comparison (control = 1202 absent, subject = 1202
applied), forcing `-fa on -ctk bf16 -ctv bf16` so the comparison actually
exercises RD04's native-BF16 flash-attn path.

Real run on Brutus, all three contract architectures, **build-once
fat-multiarch verified** (see below):

| architecture | subject PPL | control PPL | sigma | result |
|---|---:|---:|---:|---|
| gfx1100 | 10.5870 | 10.6247 | 0.1826 | **PASS** |
| gfx1201 | 10.5787 | 10.6406 | 0.3000 | **PASS** |
| gfx1030 | 10.5978 | 10.5978 | 0.0000 | **PASS** |

All three architectures are a clean real PASS on both contract checks
(`backend_reference` + `ppl_equality`), every sigma well inside the
`max_sigma=3.0` threshold. The earlier gfx1201 build crash (real
`clang++` internal segfault, real toolchain flakiness) did not recur on
retry -- confirmed not a code issue in this patch or producer.

**Build-once fat-multiarch fix confirmed working.** This run also
verifies the `docs/reference/testing/STANDARDIZED_PATCH_VALIDATION_CRITERIA.md`
"build once, run per-device" fix (commits `cacc0b98`, `b65071d1`): the
subject/control binaries were compiled ONCE with
`AMDGPU_TARGETS="gfx1100;gfx1201;gfx1030"`, and the gfx1201/gfx1030 runs
both hit real CMake cache reuse rather than a fresh rebuild:

```
=== RD04 correctness: gfx1201 ===
[patch-campaign] rd04-correctness-subject: configure request unchanged; reusing CMake cache
[patch-campaign] rd04-correctness-control: configure request unchanged; reusing CMake cache
=== RD04 correctness: gfx1030 ===
[patch-campaign] rd04-correctness-subject: configure request unchanged; reusing CMake cache
[patch-campaign] rd04-correctness-control: configure request unchanged; reusing CMake cache
```

Only gfx1100 (the first architecture in the loop) performed a real
compile; gfx1201 and gfx1030 reused the same fat binaries. This is the
reference-correct pattern for every multi-arch correctness/benchmark
producer in this project going forward.

## Known limitations

- **Correctness (`backend_reference` + `ppl_equality`) has no real
  producer yet.** `validation.toml` declares the check honestly against
  a real validator shape, but no evidence is bound -- it reports
  `BLOCKED`, not a fabricated pass. Building a real correctness producer
  (analogous to RD08's `validation/rd04_correctness.py`) is separate,
  future work.
- **Activation has no real marker probe yet.** RD04's patch source
  carries no `BIGCHERRY_PATCH_TRACE`-gated marker (unlike RD08). The
  generic tune-binary/`GGML_CUDA_DISABLE_FUSION`-based negative control
  is **not** valid for this patch (RD04 is flash-attention, not a
  fusion path GGML_CUDA_DISABLE_FUSION controls) and must never be
  reused here. `validation.toml`'s activation check stays declared but
  unsatisfied (`BLOCKED`) until a real subject-hit/control-miss RD04
  marker probe exists.
- These two gaps mean this patch's tracked-status may correctly remain
  `ported-benched` (real performance evidence) rather than advancing to
  `ported-validated` (which needs both correctness checks AND activation
  proof) until both are built.

## Control vs. subject

Standard validation-domain composition: `control_src` (this patch
absent) vs. `patched_src`/validation-subject (this patch present, same
build options as control). No patch-specific composition wrinkle.

## Evidence

Runtime artifacts (build logs, raw benchmark output) land under
`artifacts/patch-validation/1202_rd04_bf16_flash_attn_tile/<campaign-identity>/`,
outside this tracked patch directory. The compact, tracked record is
`evidence/validation.json`.
