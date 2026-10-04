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

Run correctness, scheduling and performance gates on the real BigCherry Flash-Next topology. Promotion requires target/MTP device overlap and critical-path reduction, not merely successful ahead generation.

## Steps

1. Build paired arms from one `patch-refactor` baseline:
   - control: MTP, ahead=0;
   - fixed ahead;
   - adaptive ahead.
2. Hold model/sidecar metadata, target split, KV types, batch/ubatch, MTP GPU, front depth, sampler and unrelated BigCherry toggles constant.
3. First lane: `--parallel 1`, temperature 0, fixed front depth; then 1268 adaptive front depth.
4. Test shallow ~8-10K and deep ~65-80K contexts plus long steady decode.
5. Include code, structured/repetitive text, prose and deliberately low-acceptance output.
6. Separate round classes in telemetry:
   - cold/flush round with serial fresh front + live continuation;
   - promoted-hit round with no serial fresh front + forced-front replay/continuation under target verification.
7. Capture:
   - tok/s and ms/generated token;
   - full-front and bridge-match rates;
   - promoted/flushed tokens;
   - target `llama_process` submit host time;
   - target device interval and final `llama_synchronize` wait;
   - serial fresh-front draft time;
   - promoted-front replay time;
   - continuation time;
   - total MTP overlap work and overhang;
   - authoritative `common_speculative_process()` replay/reseed time;
   - GPU utilization/power and rocprof timeline.
8. Correctness: exact greedy target IDs vs control, no stale/frontier acceptance, reset/stop/context-shift/replay stress, long-run assert/memory scan.
9. Compatibility: 1268 off/on; 1308 off/on; other Flash-Next recipe patches identical between arms.
10. Promote only from paired/ABBA evidence with activation markers and profiler proof.

## Detailed Solution & Technical Design

Primary metrics:

```text
pipeline_hit_rate = promoted_tails / ahead_attempts
bridge_match_given_full = bridge_matches / full_front_accepts
hidden_fraction = 1 - overhang_us / max(mtp_overlap_us, 1)
critical_path_delta = control_ms_per_token - subject_ms_per_token
```

For the worker-free design, verify directly:

```text
target submit
  -> target kernels continue running
  -> MTP replay/continuation kernels run on separate GPU
  -> target sync waits only for remaining target work
```

A reduced target sync wait that matches the overlapped MTP interval is stronger evidence than a reduced host-sync count.

### Promotion gates

Absolute:
- zero greedy token divergence;
- zero accepted stale/frontier-mismatched result;
- zero draft-context lifetime/rollback failure;
- ahead=0 unchanged.

Performance:
- paired CI excludes material regression;
- >=3% median gain is a useful first screen, not a universal threshold;
- target submit/sync and profiler timeline explain the observed gain;
- non-speculative controls do not regress >1% without an explained artifact.

### Failure interpretation

- `llama_process` does not leave target work in flight: reject worker-free FMTP03 before adding complexity; only then evaluate a worker fallback.
- Promoted-front replay exceeds target verify window: reduce ahead depth or reject steady-state chaining.
- High bridge mismatch: adaptive controller should turn off; never weaken bridge rule.
- Good hit rate but worse target time: diagnose PCIe/host/thermal contention.
- Win only on predictable prompts: adaptive mode may still qualify if probe/hysteresis behavior disables elsewhere.

## Files

- experiment contract/recipe evidence
- `tools/lab/flash-next/*` helpers if needed
- plan notes after hardware runs

## Validation

Before hardware:

```bash
python docs/planning/active/patching-flash-next-mtp-pipeline/mock_pipeline.py
```

Then repository-standard patch lint/rebase tests and paired hardware runner. Record BigCherry commit, llama.cpp pin, ROCm version, device topology, model/MTP identity and full server args.

## Acceptance Criteria

- Exact greedy output.
- Profiler proves concurrent target/MTP kernels on disjoint devices.
- Promoted rounds visibly use forced-front replay under the target window, not stale speculative KV.
- Telemetry explains hit/flush/overhang and throughput delta.
- Ahead=0, 1268 and 1308 compatibility lanes pass.
- Default remains off until evidence supports promotion.
