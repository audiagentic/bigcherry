---
id: FMTP01
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

# Instrument and fail-closed gate the MTP overlap lane

## Description

Establish the exact timing/device baseline and capability gate for verification-overlap without changing scheduling. This item exists to prove that the production Flash-Next lane has an overlap window worth exploiting and that native MTP is actually isolated on a separate device.

Do not add a worker or asynchronous execution here.

## Steps

1. Add disabled-by-default configuration to `common_params_speculative_draft`:
   - `int32_t mtp_ahead = 0;`
   - CLI `--spec-mtp-ahead N`
   - env `LLAMA_ARG_SPEC_MTP_AHEAD`
   - semantics: maximum future tail tokens; bridge is additional and not counted in `N`.
2. Add a read-only MTP capability query instead of making server code inspect implementation internals. Report `is_mem_shared`, `chain_heads`, `n_mtp_layers`, probabilistic mode and effective `n_max`.
3. v1 eligibility must fail closed unless all are true:
   - speculative type is `draft-mtp`;
   - `mtp_ahead > 0`;
   - `!is_mem_shared`;
   - `!chain_heads`;
   - greedy/sample-and-match mode (`!params.probabilistic`);
   - one active sequence for the qualification path.
4. Add phase counters around the existing lifecycle:
   - front MTP draft time;
   - draft rollback/truncate time;
   - target verify time;
   - `common_speculative_process()` draft re-evaluation time;
   - target acceptance time;
   - full-front acceptance count/rate.
5. Add one activation/rejection marker containing requested ahead depth, effective eligibility and one bounded reason code (`shared_memory`, `chain_heads`, `probabilistic`, `multi_seq`, `disabled`).
6. Record target and draft backend/device names at startup. On HIP qualification, capture rocprof/rocprofv3 traces proving the MTP decode kernels execute on the intended separate card and identifying any target-GPU work caused by the draft context.
7. Keep high-cardinality timing details in trace/evidence artifacts. Only add server metrics when labels/counters are bounded.
8. Add a paired baseline contract for the exact Flash-Next deployment recipe and preserve all existing target/MTP arguments except the subject-only ahead opt-in.

## Detailed Solution & Technical Design

The exact c061 MTP class already exposes the necessary mode split internally:

```cpp
int32_t n_mtp_layers  = 1;
bool    is_mem_shared = false;
bool    chain_heads   = false;
...
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
```

Do not hard-code Qwen model names. The safety condition is execution semantics, not architecture strings.

Suggested public/common-side query:

```cpp
struct common_speculative_mtp_caps {
    bool valid = false;
    bool memory_shared = false;
    bool chain_heads = false;
    bool probabilistic = false;
    int32_t n_mtp_layers = 0;
    int32_t n_max = 0;
};

bool common_speculative_get_mtp_caps(
        const common_speculative * spec,
        common_speculative_mtp_caps * out);
```

The query should return false for non-MTP compositions rather than relying on RTTI in `server-context.cpp`.

Timing must use synchronization points that already exist. Do not insert `llama_synchronize()` solely to obtain measurements; that would perturb the very overlap opportunity being measured. Prefer host wall-time around already-synchronous phases and HIP trace for device execution.

Branch-specific constraints:

- `1293_sched_single_input_sync` already removed many host synchronizations with neutral wall time. Treat sync count as diagnostic, not the objective.
- `1280_qwen4exp_mtp_kpool_alloc` is rejected. Eligibility assumes corrected MTP sidecar metadata, not that patch.
- `1268_prbe52_adaptive_mtp_wiring` owns front-depth adaptation; this plan must not add a competing front-depth knob.

## Code Samples & Guidance

Argument validation:

```cpp
if (value < 0 || value > 32) {
    throw std::invalid_argument("--spec-mtp-ahead must be in [0, 32]");
}
params.speculative.draft.mtp_ahead = value;
```

Fail-closed reason helper should be pure and testable:

```cpp
enum class mtp_ahead_reason {
    eligible,
    disabled,
    not_mtp,
    shared_memory,
    chain_heads,
    probabilistic,
    multi_seq,
};
```

## Files

- `common/common.h`
- `common/arg.cpp`
- `common/speculative.h`
- `common/speculative.cpp`
- `tools/server/server-context.cpp`
- optional bounded server metrics files
- new patch package + patch mechanics tests
- `config/experiment-contracts.toml`

## Validation

Offline:
- argument bounds/default tests;
- capability-query tests for all three MTP modes and non-MTP;
- `mtp_ahead=0` never changes draft scheduling or creates new threads;
- patch apply/idempotence/missing-anchor tests;
- trace marker appears only when requested.

Hardware:
- production Flash-Next target tensor split + separate MTP GPU;
- fixed seed/temp/depth, at least 10 steady-state rounds;
- rocprof device lanes for target verify and MTP draft;
- collect phase distributions at shallow and deep context.

## Effort & Risk

M / low. It is instrumentation plus capability plumbing. Primary risk is measurement perturbation from accidental synchronization.

## Acceptance Criteria

- Ahead defaults to off and existing output/performance is unchanged within noise.
- Production lane is explicitly classified eligible or rejected with one concrete reason.
- Device trace proves where native MTP work actually executes.
- Baseline data is sufficient to estimate `T_front`, `T_verify`, replay time and available overlap window.
- No new synchronization is introduced just for telemetry.
