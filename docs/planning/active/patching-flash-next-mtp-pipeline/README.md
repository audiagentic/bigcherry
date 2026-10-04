# Flash-Next MTP verification-overlap pipeline

Plan set for hiding native MTP draft work under target verification on BigCherry's separate-draft-GPU Flash-Next topology.

## Baseline and scope

- BigCherry branch: `patch-refactor` (design revalidated from `af3898e8d98afe9c4de2b5813394c19d0fce325d`).
- llama.cpp pin: `c061df198`.
- Target workload: Qwen4Exp / Flash-Next with target tensor split and a separate MTP GPU.
- This is a scheduling/state-machine optimization. It does not change target acceptance semantics.
- QFP13 remains the owner of same-step decode launch-count reduction. This plan targets a different seam: overlap between target verification and future MTP work.

## Target transformation

Current server critical path, simplified:

```text
draft A
-> restore/truncate draft context
-> target verifies A
-> common_speculative_process() re-evaluates verified rows in ctx_dft
-> target accepts/rejects
-> next round drafts again
```

Proposed bounded pipeline:

```text
draft front A
-> launch MTP continuation (bridge + B) ───────────┐
-> target verifies A                              │ concurrent, separate GPU
-> join continuation <─────────────────────────────┘
-> preserve authoritative draft-context replay/reseed
-> target accepts/rejects A
-> promote B only if full-A acceptance + bridge equality
```

The target remains authoritative for every emitted token. Ahead tokens are only proposal work and are always verified normally before they can be emitted.

## Why native MTP can continue ahead

At llama.cpp `c061df198`, `common_speculative_impl_draft_mtp` has three modes:

- shared-memory assistant (`is_mem_shared`),
- chained trained heads (`chain_heads`),
- single-head non-shared MTP (Qwen path).

For the single-head non-shared path, the first input is `(dp.id_last, pending_h)`. After each draft decode the implementation samples `id`, reads `h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last)`, and feeds `(id, h_row)` to the next MTP position. Therefore, once front A has been generated, the MTP context has all state needed to continue its own speculative branch without another target hidden-state row.

`accept()` later overwrites `pending_h` from `verify_h[n_accepted]`; that remains the authoritative reseed path after target verification.

## Bridge rule

A continuation cannot be reused merely because all front tokens were accepted.

If the current draft is:

```text
A1 A2 ... AN
```

then continuing MTP produces:

```text
bridge B1 B2 ... BM
```

while a full target verification produces:

```text
A1 A2 ... AN Z
```

The continuation tail is on the authoritative prefix only when both are true:

1. target accepts all `N` tokens in A;
2. `bridge == Z`.

The bridge is consumed by this equality check. Only `B1..BM` may become the next verification draft. Partial acceptance, bridge mismatch, stale epoch, reset, replay, context shift, failure, stop/EOS, or frontier mismatch flushes the ahead result.

## Compatibility envelope for v1

| Mode | v1 | Reason |
| --- | --- | --- |
| native MTP on separate draft context/device | yes | intended independent execution lane |
| single-head (`!chain_heads`) | yes | self-continuing hidden-state recurrence is proven at c061 |
| non-shared memory (`!is_mem_shared`) | yes | avoids concurrent mutation of target-shared memory |
| greedy / sample-and-match | yes | deterministic first qualification lane |
| `--parallel 1` | yes | simplest lifecycle and epoch proof |
| probabilistic rejection | later | proposal distributions + sampler state must be exact |
| chained trained heads | later/no | head-indexed KV semantics differ |
| shared-memory assistants | no | unsafe to mutate shared state concurrently |
| multi-slot | later | requires independent epochs/jobs per sequence |

## Existing BigCherry work this set must reuse

- `1254_nro05_gdn_mtp_prefix_tail`: recurrent/GDN MTP correctness surface.
- `1255_nro06_adaptive_mtp_depth`: pure adaptive depth controller.
- `1261_nro10_spec_ctx_other_devices`: speculative scheduler visibility across target/draft devices.
- `1268_prbe52_adaptive_mtp_wiring`: runtime per-sequence adaptive MTP depth and acceptance accounting; includes the current `n_min_adaptive >= n_min` guard.
- `1280_qwen4exp_mtp_kpool_alloc`: rejected; do not depend on it. Correct Flash-Next MTP metadata is required instead.
- `1293_sched_single_input_sync`: evidence that host-sync count reduction alone was neutral; do not duplicate this direction.
- `1308_qwen4exp_rollback_copy_no_cont`: may reduce same-step rollback snapshot launches; pipeline must remain correct with either 1308 arm.

## Dependency graph

```text
FMTP01 instrumentation + fail-closed eligibility
  -> FMTP02 synchronous continuation primitive
  -> FMTP03 persistent async worker / overlap
  -> FMTP04 bridge-validated tail promotion
  -> FMTP05 adaptive ahead-depth economics
  -> FMTP07 hardware qualification

FMTP04 -> FMTP06 probabilistic + multi-slot extension -> FMTP07
```

FMTP06 is not required for the first greedy single-slot hardware proof.

## State invariants

1. Target sampling/verification is the sole commit authority.
2. Every ahead job/result carries `(seq_id, epoch, base_pos, front_len)`.
3. Stale epochs/frontiers are rejected, never repaired heuristically.
4. Tail promotion requires full front acceptance **and** bridge equality.
5. A promoted tail is still a normal speculative draft and must be target-verified.
6. `ctx_dft` has a single owner while ahead work is running; `common_speculative_process()`, reset/checkpoint restore and destruction cannot race it.
7. Existing replay/rollback and `accept()` hidden-state reseed remain authoritative after each verification.
8. `ahead=0` must preserve the existing path and performance within noise.

## Mock oracle

`mock_pipeline.py` models front verification, target-extra token, bridge-gated tail reuse, epoch invalidation and stale completion rejection. It is deliberately not a KV/backend simulator.

Design qualification run:

```text
PASS trials=5000 promotions=64199 flushes=316763 stale_rejected=316763
```

The randomized oracle reproduces the target truth stream exactly for draft depths 1..8 and varied synthetic draft accuracy. Real MTP hidden/KV equivalence remains an FMTP02 hardware/mechanics gate.

## Primary performance question

Do not promote this feature based on "ahead hit rate" alone. The relevant result is critical-path reduction:

- target wait on MTP work,
- join overhang,
- target verify ms/step,
- end-to-end generation latency / tok/s,
- and device contention while target and MTP GPUs overlap.
