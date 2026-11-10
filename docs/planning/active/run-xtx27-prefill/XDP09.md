---
id: XDP09
order: 8
plan: run-xtx27-prefill
state: active
created-at: '2026-11-09T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Integrate verified XDP optimization slices without production regressions

## Objective

Bring XDP's first **reviewable, independently mergeable** work to
`main`, then resume bounded optimization of the existing Radiance
gfx11 INT8/MXFP4 and MoE kernels. This plan is a release and
validation ledger, not a claim that isolated INT4 results automatically
improve production GEMMs. Use the existing `exp/xtx27-prefill-opt`
worktree; do not create another worktree or modify other agents'
RAD49/RAD50 branches. The GPU lease belongs to whichever agent
holds it; never launch an unleased test or serving process.

## Decomposition and merge order

### Slice A: XDP01–XDP02 prefill and KV budget harness

- Files: `xtx27_prefill.py`, `xtx27_kv_budget.py`, tests,
  XDP01–02 plans, lab progress ledger, required tooling disposition
  registry additions.
- Status: [PR #784](https://github.com/audiagentic/bigcherry/pull/784)
  was opened on 2026-11-09. The source tree was verified byte-for-byte
  against the original `f5fc972b3` commit after uploading seven files.
  A required CI boundary check found that newly tracked lab files
  must be recorded in `docs/reference/tooling/TOOL_DISPOSITION.md`.
  The PR branch was amended to register all three new lab files;
  title CI uses `test(radiance): ...` because `exp` is not an allowed
  PR title type. **Merged in main** via PR #784 at commit `45f6f6294676c067784a9649862b349c2411dc25`, with both required checks passing.
- Acceptance: full PR CI green, no changed serving kernel or production
  settings, squash into `main`, verify merged SHA. This slice provides
  reproducible test infrastructure; **it is not itself a measured speedup**.

### Slice B: XDP03 quantization planner and selector

**Merged in main:** [PR #786](https://github.com/audiagentic/bigcherry/pull/786), squash commit `a081ddd801fafd035249c42cb51f71fbc96434e2`, both PR CI and title green. The CI gate additionally required one category-E generic filename declaration in `model_constant_guard.toml` and five disposition entries for RAD42 lab scripts newly merged concurrently. No production code or inference defaults changed.


- Files: `xtx27_quant_plan.py`, `xtx27_quant_select.py`, their
  card-free tests, XDP03 plus small XDP01/02 progress updates, and
  the two additional `TOOL_DISPOSITION.md` rows.
- Acceptance: feature remains a lab planner, does not convert the
  original BF16 checkpoint or choose a production quantization mode.
  Run relevant card-free tests and entire offline CI, then merge as
  a separate small PR. Source blob SHA must match the recorded local
  revision; no shortcuts around required checks.

### Slice C: XDP04–XDP07 native INT4 experiments

- Standalone gfx1100 IU4 W4A4 research probes, coalesced output
  epilogue, 16-wave metadata reuse, per-group quality diagnostics,
  source/binary provenance, and XDP04–07 plans.
- **Measured** 1,057 us kernel-only for the staged-meta, coalesced
  128×128 kernel at M512/N17408/K5120 versus ~928 us MXFP4
  on the same XTX. Precision/producer/real-model quality differ,
  so do not wire IU4 into production. Merge test tools in
  bounded research-only PR(s) after full CI/disposition bookkeeping.
  No `vendor/radiance` changes.
- Do not merge or deploy experimental code if it lacks source
  hashes, independent oracle checks or a clear opt-in.

### Slice D: XDP08 native INT8 W-scale staging

- Existing native gfx11 `r4d_gemm_8a8_prefill.h.rw` only, behind
  default-OFF `R3_I8_SWS_LDS`. The source uses two 128-entry
  alternating f16 LDS metadata buffers so the next K128 group cannot
  overwrite scales still being consumed; all original arithmetic,
  native IU8 WMMA, quantization and output contracts remain intact.
- Card-free address bijection, native rewrite drift check, 67 XDP
  tests and docs tests have passed. An isolated low-priority gfx1100
  CMake build is attempted in scratch, independent of the active
  engine. Nothing is an INT8 **speedup** until numerical and matched
  card tests demonstrate it.
- **Merge gate**: opt-in variant compiles to correct native gfx1100
  ISA and has no disastrous register/spill regression; checked
  `rad-kbench` against unchanged implementation and libref on the
  original `q27-w8a8-tok2048.rkb` fixture; repeated same-card
  A/B including M64/128/512/2048 while holding GPU lease; no
  regressions on FP8/RB/IQR; then promote only as a default-OFF
  experimental flag. Production default ON additionally requires
  end-to-end throughput, BF16-origin KLD and TP2 consistency.
- **Decision:** the earlier direct INT4 1.85× kernel gain is
  **not** a credible INT8 gain prediction, since INT8 already
  stages activation scales and software-pipelined weight tiles.

### Slice E: native MXFP4 and MoE follow-ons

- No immediate generic “stage more scales” change. MXFP4 already
  has `sMag` and `sR`, and MoE `s_as`/`s_ws`.
- Profile the actual generated gfx11 ISA for weight expansion,
  bf16 widening, per-row scale carry and output stores; measure
  repeated load/use and VGPR live ranges per projection.
- MoE: separate route/sort/dispatch, expert compute and
  remote-helper/TP2 communication; previous Flash-Next profile
  shows dominant `moe_gemm_q` but that is an E4M3/nonlinear
  codebook path, not interchangeable with signed IU8.
- Require *independent* A/B matched row/routing/shape, oracle and
  BF16-origin quality before modifying production defaults.

## Operational contract

- GitHub PRs require conventional titles
  `feat|fix|perf|chore|docs|test|refactor|ci|build|bump`.
- Any new tracked `tools/lab` file must appear exactly once in
  `docs/reference/tooling/TOOL_DISPOSITION.md`; the baseline
  inventory is immutable.
- The lab host has no HTTPS Git push credentials. Publish distinct,
  reviewed slices via the connected GitHub API using source
  blob hashes and a base-tree-relative commit. Reject truncated
  uploads and stale branches. Do not bypass required CI checks.
- One existing code worktree, original radiance pin read-only,
  no change to GPU defaults or live services, and no unleased GPU.
- Maintain a true distinction between `implemented`,
  `card-free tested`, `compiled`, `GPU numerically validated`,
  `benchmarked`, `merged`, and `production enabled`.

## Evidence ledger

- Local experimental source branch HEAD at XDP08 start:
  `9d6a20008`, followed by `5eea83934` fixing the
  alternating-group scale reader/writer race.
- Initial PR #784 branch title failed conventional-title CI,
  and its first GitHub upload had two silently shortened
  Python files. Uploaded replacements were verified against
  their original blob SHAs, the branch ref was replaced with
  a hash-verified commit, and its title corrected. The next
  CI failure was the required tooling disposition inventory;
  added three entries on the PR branch and reran CI.
- No measured speed benefit from XDP08 yet. Preserve the
  currently shipping INT8/MXFP4/MoE control implementations
  until those gates pass.

## XDP08 build compatibility and GPU lease gate

The initial source37 HIP `libr3` builds completed, but numeric launches were **rejected by the kernel loader before executing a test**: the chosen rad-kbench executables expected ABI31 or ABI35. This is **not** numerical failure or speed evidence. The pinned, matched ABI35 pair for XDP08 is source `${BC_WORK_ROOT}/sources/98156549a09fba19e384a8c9decfe345` and runtime prefix `${BC_WORK_ROOT}/builds/ccb649ea788a5272eead885012a9a915/a4040b3a1a4f4cd9002bfa7c081e2ddd/runtime-prefix`; that source passes the complete native rewrite drift check with zero failures. A cached `radiance_DIR` initially resolved an older ABI31 prefix despite the new `CMAKE_PREFIX_PATH`; both ON and OFF experimental scratch builds were explicitly reconfigured with the **matching ABI35 radiance_DIR**, with no modification to the serving build or production runtime. No unleased GPU runs are permitted. As of this checkpoint a concurrent BigCherry 4-GPU regression process holds the exclusive lease, so actual ON/OFF checked numerical and speed measurements remain a separate, mandatory gate.

**Decision XDP09-D1:** do not call ABI-rejected tests regressions and do not land/enable an unmeasured INT8 kernel optimisation. Pair the correct ABI35 plugin with the same ABI35 `rad-kbench` and `libref`, and compare ON/OFF against identical fixtures. All GPU and quality results must be new, not inherited from the standalone IU4 results.
