---
id: QFP07
order: 0
plan: patching-qwen-flash-next
state: pending
created-at: '2026-10-03T15:20:52.461503+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# 1303 separate attention/KV tensor split (BIGCHERRY_ATTN_TS / _ROTATE)

## Description

Patch `1303_attn_kv_tensor_split` decouples Qwen4Exp full-attention/KV placement from the model-wide tensor split. Under `-sm tensor`, `BIGCHERRY_ATTN_TS` gives the full-attention family its own split vector; `BIGCHERRY_ATTN_ROTATE=0` keeps whole KV heads on chosen GPUs. Recurrent/GDN layers that reuse attention-named tensors retain normal `-ts`; indexer tensors/caches remain mirrored.

The next optimisation slice is context-aware attention placement, followed only if justified by a tightly gated decode sequence-sharding experiment. QFP09 profiling shows memory-optimal `1,1,0` placement leaves the R9700 waiting while XTX flash-attention time grows sharply with context. Qwen4Exp has only two KV heads, so head-based placement reaches a granularity floor before all three GPUs can contribute unique KV capacity.

QFP07 is the single owner for attention/KV placement and any QSA sequence partitioning policy. QFP09 owns generic rank-lateness diagnosis and QFP13 owns launch/fusion work; neither should grow a second attention-placement mechanism.

## Steps

1. Preserve/revalidate the `1,1,0` memory-optimal baseline at 10K/80K/160K/192K/228K/240K.
2. Build a context-length cost model from per-rank FA time, KV bytes, free VRAM, allreduce wait and end-to-end token time. Test static splits that give the R9700 a measured share without violating two-head placement constraints.
3. If two stable regimes exist, add startup-time profile selection only (for example speed-optimal at moderate context, capacity-optimal at maximum context). Do not migrate KV tensors or change split while a sequence is live.
4. Diagnose the 248K prefill collapse independently: sweep 228K through 248K in small increments and capture device/host memory, paging/faults, FA kernel/geometry, workspace/scratch and per-layer latency. Do not attribute it to VRAM pressure without evidence.
5. Only if static placement cannot use the R9700 effectively and attention-induced critical-path skew remains >=1 ms/token at >=160K, prototype QSA decode sequence sharding: partition K/V token ranges, compute local online-softmax state, combine partial states.
6. Reuse existing collective/stream/event/transport infrastructure. Benchmark exact partial-state payloads through available direct/host-staged paths before implementing the full graph; abandon if communication lower bound cannot beat measured attention skew.
7. Keep prefill context parallelism out of scope until decode proves value.
8. Promotion evidence: greedy/raw-logit/KLD parity, ABBA, deep-fill capacity, per-rank timing, communication bytes/time, and maximum-context stability.

## Detailed Solution & Technical Design

### Static context-aware placement first

Model token time by the slowest rank, not average utilization:

`T_token ~= max_r(T_qsa[r] + T_other[r]) + T_collective_wait`

Record the slope of QSA time versus live context separately per GPU. Moving attention to the R9700 is useful only when it reduces the slowest-rank completion time enough to pay for extra KV residency/communication. If speed/capacity regimes are clearly separated, expose deterministic named startup profiles keyed by requested max context/KV budget; explicit environment overrides remain authoritative.

### 248K collapse attribution

The observed ~1100 -> ~85 pp t/s discontinuity at 248K is not normal smooth long-context bandwidth scaling. Sweep 228K/232K/236K/240K/244K/248K and capture allocated/free bytes, host RSS/commit/page faults, FA kernel/launch geometry, workspace allocation, per-QSA-layer kernel/gap time, and whether onset follows a token boundary, allocation threshold or particular layer. Fix the attributed mechanism directly rather than sacrificing placement.

### Decode sequence-sharding gate

For one query/head, each rank over its local token interval computes online-softmax state:

```text
m_r = max(scores_r)
l_r = sum(exp(scores_r - m_r))
o_r = sum(exp(scores_r - m_r) * V_r)
```

Combine stably:

```text
m = max_r(m_r)
l = sum_r(l_r * exp(m_r - m))
o = sum_r(o_r * exp(m_r - m)) / l
```

This communicates O(head_dim) partial state rather than gathering K/V proportional to context. On this no-P2P workstation, transport economics are the primary gate. Measure direct and host-staged small-state exchange first. Do not materialize local probability vectors for reduction, create a second communicator, or add a new scheduler.

The prototype is Qwen4Exp/QSA-only. GDN/recurrent layers and indexer caches remain untouched. Architecture checks after the Gemma-4 corruption finding remain fail-closed.

## Code Samples & Guidance

Pseudo-shape only:

```cpp
if (arch == LLM_ARCH_QWEN4EXP && qfp07_seq_shard_decode && is_qsa_layer(il)) {
    kv_range range = shard_sequence(kv_len, rank, world);
    fa_partial p = flash_attn_partial(q, k_cache, v_cache, range);
    fa_partial total = reduce_online_softmax_state(p, existing_comm, stream);
    write_attention_output(total);
}
```

Prefer exposing/factoring the existing FA online-softmax accumulator. A new partial-attention kernel is justified only if the current kernel cannot expose state without extra global-memory traffic.

## Files

- patch `1303_attn_kv_tensor_split`: Qwen4Exp attention/KV classification and startup placement policy
- model/context graph construction: QSA-only sequence-shard graph if gate passes
- existing HIP collective/event/transport abstraction: small partial-state combine
- benchmark tooling: context sweep, per-rank FA timing, memory/paging and communication payload/time
- tests for architecture fail-closed behavior, profile selection and online-softmax combine parity

## Validation

Required lanes: `1,1,0` baseline plus at least two R9700-sharing static candidates; 10K/80K/160K/192K/228K/232K/236K/240K/244K/248K where capacity permits; production f16/f16 and f16/q8_0 KV configurations; sequence-shard off/on only after static gate; 2-rank XTX and 3-rank candidates where transport allows; pp512/2048 and tg128/512 with production MTP.

Capture pp/tg, per-rank QSA time, collective wait, sequence-shard reduction time, bytes/token, kernel count, busy/idle, peak VRAM, host RSS/commit/page faults and max usable context.

Correctness: greedy identity where expected plus raw-logit/KLD against non-sharded reference. Test very negative scores, one dominant rank, empty/tail shards and non-divisible context lengths.

Startup profile promotion: >=3% decode improvement in intended context band, no representative lane >2% slower, no loss of supported max context versus replaced profile.

Sequence-shard promotion: >=5% end-to-end decode at >=160K on at least two long-context lanes, communication+combine <=50% of attention time saved, no enabled lane >2% slower, correctness unchanged. Otherwise retain static split.

## Effort & Risk

Static placement/248K attribution: medium. Sequence sharding: high and explicitly conditional. Risks are communication latency, numeric combine correctness, graph synchronization and duplicate collective ownership.

## Standards

- QFP07 is the single attention/KV placement owner.
- QFP09 remains generic arrival-skew owner; QFP13 remains launch/fusion owner.
- Reuse existing collective/stream/event/transport abstractions.
- Qwen4Exp-only until another architecture has explicit grouping/parity proof.
- Prefer measured startup profiles over opaque runtime heuristics.
- No live KV migration.
- Fail closed to existing static attention path.

## Acceptance Criteria

- 248K collapse has evidence-backed attribution or an explicit unresolved measurement gap.
- Static R9700-sharing candidates are compared with `1,1,0` by slowest-rank critical path, capacity and correctness.
- Stable speed/capacity regimes, if present, use deterministic overrideable startup selection.
- Sequence sharding is attempted only after critical-path and transport gates pass.
- Sharded path uses online-softmax partial-state reduction, not KV gathering, and reuses existing communication infrastructure.
- Greedy/logit/KLD, tail-shard, long-context and max-capacity gates pass.
- Gemma/non-Qwen4Exp remains rejected by experimental placement/sharding.

## Notes

Background: with 2 KV heads and 3 GPUs the default split rotates heads so each GPU holds one head for 8 of 12 QSA layers, coupled to expert-weight placement. KV on both XTX (`1,1,0`) loads to 248K and is usable at 240K; 248K prefill collapses ~1100 -> ~85 t/s and 256K OOMs. Earlier profile ABBA showed material pp/tg gains and deep fills to 164K/228K pass.

2026-10-04 load-balance finding: with attention on two XTX, XTX FA grows ~0.23 -> 1.5 -> 3.6 ms/token at 10K/80K/160K while R9700 waits in allreduce at long context. This is the trigger for static placement sweep and conditional sequence-shard gate.

2026-10-04 cross-model safety: Gemma-4-26B-A4B produced garbage with `BIGCHERRY_ATTN_TS=1,1,0`; 1303/1305 now fail closed outside qwen4exp. Flash-Next under 1303 was previously bit-identical.

2026-10-05, folded in from BCOP19 (audit backfill) - sequence-sharded long-context attention gate: pursue only if online-softmax partial-state exchange over the no-P2P topology beats measured XTX attention skew; benchmark exact partial-state transfer before full implementation and promote only for >=5% long-context decode on two lanes with communication <=50% of attention time saved and correctness passing.

## Change Log

- 2026-10-03T15:20:52.461503+00:00 (created-by): Created by agent
- 2026-10-03T16:05:30.647552+00:00 (updated-by): Updated: section:notes
- 2026-10-03T23:52:04.938996+00:00 (updated-by): Updated: section:notes
- 2026-10-05: BCOP19 audit backfill added sequence-sharding gate.
- 2026-10-05: Transplanted context-aware placement, 248K attribution, online-softmax sharding design, validation and ownership details from `automation-qfp-indexer-20261004`.

## Ledger-events

- chg_20261003_235207_the-flash-next-only-gpu-split_2755
- 2026-10-03T23:52:10.433314+00:00 (updated-by): Updated: section:ledger-events
- 2026-10-05T04:54:56.316602+00:00 (updated-by): Updated: section:notes
