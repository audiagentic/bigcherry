---
id: PHC03
order: 0
plan: patching-hip-collectives
state: pending
created-at: '2026-09-10T02:11:29.105053+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Fix META segfault for D=3+ heterogeneous no-P2P groups (ggml_backend_cuda_cpy_tensor_async)

## Description

RU01's real-hardware D=3 heterogeneous correctness qualification found META (GGML_HIP_REDUCE_PLAN=meta) segfaults (SIGSEGV) rather than falling back gracefully, when a SPLIT_REDUCE graph spans 3+ devices with no PCIe P2P access. HI84 already proved META works correctly for D=2 no-P2P pairs on this exact hardware; the fold/copy-back path apparently has an assumption that breaks specifically at N>2.

## Steps

1. Reproduce: build the current test-hip-reduce probe, run `GGML_HIP_REDUCE_PLAN=meta test-hip-reduce --case <any D=3 case> --plan meta --devices 0,1,3 --out result.json` on a no-P2P 3+ GPU box -- crashes with SIGSEGV.
2. Get a symbol-resolved backtrace (gdb -batch -ex run -ex bt) into ggml_backend_cuda_cpy_tensor_async / ggml_backend_meta_graph_compute's fold/copy-back lambda -- the observed crash is inside opaque HIP driver code (libamdhip64.so, no symbols) called from that copy path, so isolate exactly which copy (host-staged fallback vs peer-direct) is being attempted for device index >1 in a no-P2P group.
3. Determine whether the fold/copy-back logic assumes a fixed D=2 topology (e.g. a single pairwise host-staging buffer reused incorrectly for a 3rd participant) or whether it's a genuine missing-fallback case for N>2 no-P2P.
4. Fix the root cause -- likely in ggml_backend_meta_graph_compute's copy-back lambda or the cross-device cpy_tensor_async call it drives.
5. Re-run RU01's D=3 correctness qualification (same driver/devices/cases) to confirm the fix -- both {0,1,3} and {0,1,2} must pass cleanly before this closes.
6. Only then re-attempt D=4 qualification (RU01's own next stage, blocked until this fix lands and D=3 passes).

## Detailed Solution & Technical Design

Real crash evidence (2026-09-10, Brutus, build_plan_id 0bfc43c12b47fd79b26a80c14b656b72, AMDGPU_TARGETS gfx1100;gfx1201;gfx1030):

Thread 1 "test-hip-reduce" received signal SIGSEGV.
#0 in ?? () from /opt/rocm-7.2.4/lib/libamdhip64.so.7
#1-#4 more opaque libamdhip64 frames
#5 ggml_backend_cuda_cpy_tensor_async(ggml_backend*, ggml_backend*, ggml_tensor const*, ggml_tensor*)
#6 ggml_backend_tensor_copy_async
#7 ggml_backend_meta_graph_compute(...)::$_4::operator()(...)::{lambda}::operator()
#8 ggml_backend_meta_graph_compute(ggml_backend*, ggml_cgraph*)
#9-#11 scheduler/main

Reproduced twice on two different D=3 device sets ({0,1,3} and {0,1,2}), both segfault identically -- not a one-off/flaky hardware issue. AUTO's RCCL arm for the same D=3 groups hits the ALREADY-KNOWN separate RCCL SIGABRT (device 3/RX 6900 XT incompatible with RCCL/NCCL) -- that is not this item's concern, already tracked as a permanent RCCL exclusion.

## Code Samples & Guidance



## Files

src/ggml/src/ggml-cuda/ggml-cuda.cu (ggml_backend_cuda_cpy_tensor_async, referenced in the crash's own error path at line ~1088 for the separate RCCL issue -- confirm exact cpy_tensor_async location); wherever ggml_backend_meta_graph_compute's fold/copy-back lambda lives (likely ggml-backend-meta.cpp per HI84's own file references).

## Validation

Fixed build must pass RU01's D=3 META qualification cleanly (no crash, numerically correct output within tolerance) on both {0,1,3} and {0,1,2} device sets, using the same tools/bigcherry/tuning/reduction.py-based driver RU01 used. Do not claim the fix without re-running real hardware evidence -- this is a segfault fix, not a design change; a null/inconclusive result is not acceptable here, only clean pass or a documented different real failure.

## Effort & Risk



## Standards



## Acceptance Criteria

META SPLIT_REDUCE completes without crash for D=3 heterogeneous no-P2P groups, numerically verified correct against the CPU-double oracle, on real hardware. D=4 qualification (RU01's blocked next stage) may only proceed after this closes.

## Notes

Filed from RU01's real-hardware finding (2026-09-10) rather than attempted as part of RU01 itself -- RU01 is explicitly run-only/correctness-evidence scope ('not a new collective implementation'), this is real source-level patch work belonging to patching-hip-collectives per this project's Run-vs-Patch scope separation (matches RU01's own notes: 'Any implementation or patch change belongs under patching-hip-collectives, not this Run item').

## Change Log

- 2026-09-10T02:11:29.105053+00:00 (created-by): Created by agent
- 2026-09-10T02:11:53.757768+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260910_021202_ran-a-real-multi-gpu-correctne_4533
- 2026-09-10T02:12:02.089512+00:00 (updated-by): Updated: section:ledger-events
