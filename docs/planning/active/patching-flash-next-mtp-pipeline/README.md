# Flash-Next MTP verification-overlap pipeline

Plan set for hiding native MTP work under target verification on BigCherry's separate-draft-GPU Flash-Next topology.

## Baseline and scope

- BigCherry branch: `patch-refactor`.
- llama.cpp pin: `c061df198`.
- Target workload: Qwen4Exp / Flash-Next with target tensor split and a separate MTP GPU.
- Target sampling/verification remains authoritative; ahead work only creates future proposals.
- QFP13 still owns same-step launch-count reduction. FMTP owns cross-device scheduling/overlap.

## Review correction: no worker in v1

At c061, llama decode submits graphs with `ggml_backend_sched_graph_compute_async()`. The server then explicitly calls `llama_synchronize(ctx_tgt)` after `llama_process(ctx_tgt, ...)` when outputs are needed. Therefore v1 does not need a background thread to overlap separate GPUs.

Preferred host schedule:

```text
llama_process(ctx_tgt, verify_batch)   # async GPU submit; do not sync yet
run bounded MTP work on ctx_dft        # same host thread, separate GPU
llama_synchronize(ctx_tgt)             # waits only for target remainder
rollback/truncate ctx_dft
common_speculative_process(...)
post_decode target accept/reject
```

This gives the same ideal device overlap as a worker (`max(T_target, T_mtp)`) without cross-thread llama/backend access, mailbox races, worker lifetime, or unload/reset joins. A worker becomes a fallback only if hardware traces prove target submission is not sufficiently asynchronous on the production HIP path.

## Review correction: promoted fronts must be replayed through MTP

The old draft-context branch used to create an ahead tail cannot survive target verification: the existing path deliberately rolls/truncates `ctx_dft`, reprocesses the verified target batch, and reseeds MTP from authoritative target hidden state.

Therefore storing only promoted tokens is safe for target verification, but it is not enough to continue another ahead chain on the following round.

Steady-state pipeline needs two cases:

### Cold / miss round

```text
serially draft front A on MTP
submit target verify A asynchronously
continue live MTP branch -> bridge + B while target runs
sync target
rollback/reseed draft context authoritatively
accept/reject A
promote B only on full-A + bridge match
```

### Hit round with promoted front B

```text
install promoted B as target verification draft; no serial fresh draft
submit target verify B asynchronously
from authoritative MTP seed, force/replay known B tokens through MTP
continue reconstructed live branch -> bridge + C while target runs
sync target
rollback/reseed draft context authoritatively
accept/reject B
promote C only on full-B + bridge match
```

The forced replay is work, but it is moved inside the target verification window. This is the mechanism that makes the pipeline continuous after the first hit without preserving a second draft-context snapshot.

## Why native MTP can continue

At c061, single-head non-shared MTP starts a round from target `pending_h`, then after each draft decode samples `id`, reads the MTP `h_row`, and feeds `(id, h_row)` to the next MTP position.

A live front therefore has a valid continuation seam. A promoted front has lost that live speculative KV after authoritative rollback, so it must be replayed from `(id_last, pending_h)` with its known tokens forced until the live frontier is reconstructed.

The replayed MTP hidden trajectory need not equal the prior speculative trajectory or a fresh next-round proposal trajectory. That is acceptable in greedy/sample-and-match mode: proposals may differ, but every promoted token is still verified by the target before emission.

## Bridge rule

If current front is `A1..AN`, ahead work produces `B0 B1..BM`, while full target verification produces `A1..AN Z`.

Reuse is allowed only when:

1. all front tokens are accepted;
2. `B0 == Z`;
3. epoch and absolute frontier metadata match;
4. remaining tail meets the existing minimum-draft policy.

`B0` is consumed by the equality check. Only `B1..BM` becomes the next verification draft.

## v1 eligibility

- native `draft-mtp`;
- `!is_mem_shared`;
- `!chain_heads`;
- greedy/sample-and-match only;
- one active sequence;
- target and draft device sets disjoint for the production qualification lane;
- no known shared model/meta buffer dependency between target and draft contexts.

The last two are performance/lifetime guards: `!is_mem_shared` only describes MTP memory semantics; it does not prove physical device isolation.

## Existing BigCherry work to reuse

- `1254_nro05_gdn_mtp_prefix_tail` — recurrent/GDN correctness.
- `1255_nro06_adaptive_mtp_depth` + `1268_prbe52_adaptive_mtp_wiring` — front-depth policy/accounting.
- `1261_nro10_spec_ctx_other_devices` — speculative device visibility.
- `1280_qwen4exp_mtp_kpool_alloc` — rejected; do not depend on it.
- `1293_sched_single_input_sync` — sync-count reduction was neutral; do not optimize sync count as a proxy.
- `1308_qwen4exp_rollback_copy_no_cont` — same-step rollback optimization; FMTP must work with either arm.

## Dependency graph

```text
FMTP01 instrumentation + eligibility
  -> FMTP02 live continuation + forced-front replay primitive
  -> FMTP03 single-thread target-submit/MTP-work/target-sync overlap
  -> FMTP04 bridge-gated promotion + promoted-front replay wiring
  -> FMTP05 adaptive ahead economics/probing
  -> FMTP07 hardware qualification

FMTP04 -> FMTP06 probabilistic + multi-slot extension -> FMTP07
```

FMTP06 is not required for the first greedy single-slot proof.

## State invariants

1. Target verification is the only commit authority.
2. Every ahead result identifies its parent `(seq_id, epoch, base_pos, front_len)`.
3. Every round transition gets a new epoch; a promoted child is retagged to the new round.
4. Tail promotion requires full-front acceptance plus bridge equality.
5. A promoted tail remains an ordinary target-verified draft.
6. A live continuation seed is an ephemeral lease over current `ctx_dft` KV + sampler frontier; any draft-context mutation invalidates it.
7. Promoted fronts must be replayed from authoritative MTP seed before another continuation can be generated.
8. Existing rollback/replay and `accept()` target-hidden reseed remain authoritative after each verification.
9. `ahead=0` preserves current behavior.

## Mock oracle

`mock_pipeline.py` now models cold rounds, promoted-front replay, recursive draft hidden state, hidden-state mismatch versus fresh reseed, bridge promotion, stale/base faults, short tails and the single-thread overlap timing equation.

Current qualification run:

```text
PASS trials=5000 rounds=381584 promotions=39286 promoted_replays=39286 hidden_mismatch_promotions=24203 flushes=342298 stale_rejected=7566 timing_cases=10000
```

`hidden_mismatch_promotions > 0` is intentional: it demonstrates that bridge-gated tails can differ from a fresh target-reseeded MTP proposal while the target output remains exact.

The mock is still not a KV/backend simulator. FMTP02 must prove real forced-front replay and continuation on c061 before scheduling changes land.

## Primary performance question

Measure critical-path reduction, not ahead hit rate alone:

- target submit time and target sync wait separately;
- MTP replay/continuation time;
- join/sync overhang (`max(0, T_mtp - T_target)`);
- target verify ms/step and end-to-end tok/s;
- cross-device contention/power/thermal effects.
