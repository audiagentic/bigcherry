---
id: FMTP07
order: 0
plan: patching-flash-next-mtp-pipeline
state: pending
created-at: '2026-10-04T00:48:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Qualify Flash-Next MTP verification overlap on the AMD deployment topology

## Description

Run the correctness, concurrency and performance gates needed to promote or reject the pipeline on the real BigCherry Flash-Next setup. Promotion requires critical-path evidence, not merely successful ahead generation.

## Steps

1. Build paired arms from the same `patch-refactor` baseline/upstream pin:
   - control: MTP, `mtp_ahead=0`;
   - fixed ahead: FMTP04 with a fixed future-tail depth;
   - adaptive ahead: FMTP05 policy.
2. Hold constant model/sidecar metadata, target tensor split, KV types, ubatch/batch, MTP GPU, front draft depth, sampler settings and all unrelated BigCherry toggles.
3. First qualification lane: `--parallel 1`, temperature 0, fixed front depth. Then repeat with 1268 adaptive front depth enabled.
4. Run shallow and deep contexts representative of current Flash-Next profiling (roughly 8-10K and 65-80K), plus a long steady decode window.
5. Prompt classes must include predictable code, structured/repetitive output, ordinary prose and deliberately low-acceptance output.
6. Capture per round:
   - generation tok/s and ms/generated token;
   - front full-accept rate;
   - bridge-match-given-full rate;
   - ahead attempts/hits/flushes;
   - ahead generated/promoted/flushed tokens;
   - target verify time;
   - front MTP draft time;
   - ahead continuation time;
   - `common_speculative_process()` replay/reseed time;
   - join overhang;
   - target wait attributable to MTP work;
   - GPU utilization/power and a rocprof overlap timeline.
7. Correctness gates:
   - exact greedy target token IDs vs ahead-disabled control;
   - no stale-publication acceptance;
   - reset/stop/context-shift/checkpoint replay stress;
   - long-run log/assert/memory-error scan.
8. Compatibility matrix:
   - 1268 adaptive depth off/on;
   - 1308 rollback-no-cont off/on;
   - QSA gather/other active Flash-Next recipe patches unchanged between arms.
9. Regression controls:
   - MTP ahead=0 must match current server behavior/performance;
   - ordinary non-speculative `llama-bench` lane if common scheduler/server code changed;
   - confirm 1293-style host-sync count is not being mistaken for performance evidence.
10. Promote only from paired/ABBA evidence with activation markers and profiler proof of actual cross-device overlap.

## Detailed Solution & Technical Design

Primary derived metrics:

```text
pipeline_hit_rate = promoted_tails / ahead_attempts

bridge_match_given_full = bridge_matches / full_front_accepts

hidden_fraction =
    1 - join_overhang_us / max(ahead_us, 1)

critical_path_delta =
    control_ms_per_generated_token - subject_ms_per_generated_token
```

Also report wasted speculative GPU work:

```text
waste_ratio = flushed_ahead_tokens / max(ahead_generated_tokens, 1)
```

Waste is acceptable only when it remains hidden and does not create meaningful contention/power/thermal throttling. A high waste ratio with positive wall-clock value can still be rational on an otherwise idle separate card; the controller should nevertheless shut off if overhang/contention makes EV negative.

### Initial promotion gate

Correctness/stability are absolute:
- zero greedy token divergence;
- zero accepted stale result;
- zero draft-context ownership race or lifecycle failure.

Performance:
- paired CI must exclude material regression;
- use >=3% median end-to-end gain as a useful first screening target, not a hard universal requirement;
- final adoption can use the repository's standard improvement/no-regression contract if critical-path telemetry explains the effect;
- ahead=0 and non-speculative controls must not regress >1% absent an explained measurement artifact.

### Branch-specific interpretation

`1293_sched_single_input_sync` reduced sync calls substantially with neutral ms/step. Therefore FMTP07 must show overlap/critical-path reduction directly. A lower sync count alone is not acceptance evidence.

`1308_qwen4exp_rollback_copy_no_cont` has already demonstrated meaningful launch-count reduction. The pipeline should stack independently: measure both 1308 arms to ensure overlap benefit is not an artifact of one rollback implementation.

### Failure interpretation

- No visible GPU overlap: reject FMTP03 execution model before tuning controller.
- Good overlap but frequent bridge mismatch: FMTP05 should usually turn ahead off; do not weaken bridge rule.
- Good hit rate but slower target verify: diagnose PCIe/host/scheduler contention.
- Ahead completes after target verify: reduce depth or reject on this MTP GPU.
- Win only on predictable prompts: adaptive policy may still promote if mixed-workload tests prove fast disabling elsewhere.

## Code Samples & Guidance

Evidence summary should include a table per context class with:

```text
control t/s | fixed t/s | adaptive t/s | full% | bridge% | hit% |
ahead us | verify us | overhang us | target-wait delta | correctness
```

Store full profiler artifacts separately; do not embed huge traces in plan notes.

## Files

- `config/experiment-contracts.toml`
- relevant model/recipe registry only if the exact Flash-Next lane is absent
- validation producer/evidence artifacts
- plan item notes/change log after hardware runs
- `tools/lab/flash-next/*` helper scripts if needed

## Validation

Before hardware:

```bash
python docs/planning/active/patching-flash-next-mtp-pipeline/mock_pipeline.py
```

Then repository-standard patch lint/rebase tests and paired hardware runner. Record exact BigCherry commit, llama.cpp pin, ROCm version, device IDs/topology, model + MTP sidecar identity and full server arguments.

## Effort & Risk

M / high operationally. The code can be correct while the optimization loses due to continuation overhang or shared host/PCIe contention.

## Acceptance Criteria

- Exact greedy target output vs ahead-disabled control.
- Profiler visibly proves target/MTP concurrent kernel execution on separate devices.
- Telemetry explains hit, flush, overhang and measured throughput delta.
- Ahead=0 preserves current behavior.
- 1268 and 1308 compatibility lanes pass.
- Promote/reject decision is evidence-backed; feature remains default-off until this item passes.
