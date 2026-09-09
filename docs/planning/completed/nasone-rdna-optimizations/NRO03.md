---
id: NRO03
order: 3
plan: nasone-rdna-optimizations
state: superseded
created-at: '2026-09-08T09:50:40+10:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: L
---

# Correctness-first HIP P2P transport provider for internal AllReduce

## Description

Evaluate nasone commit `7c5bb5cb991670676b89cddc5077c9456c5cf70e` (`ggml: enable internal HIP all-reduce and P2P`) as an optional two-GPU transport provider on top of validated `1001_hip_internal_allreduce`, but do **not** mechanically port its fixed-issuer peer-copy design.

BigCherry has stronger hardware evidence than the source fork for the target dual-RX-7900-XTX topology: on Brutus/gfx1100, peer-copy/read direction matters. Destination-current/pull operations have produced silent success with zero/garbage data, while source-current/push copies are correct; kernel-side direct peer reads and cache visibility have separately failed. Corrected push-copy measurements show only a modest sub-256MB advantage and near parity at 1GB, while GP11 found host staging faster for its corrected direct-P2P AllReduce prototypes. Therefore this item is P0 because communication is important and the source provides a concrete provider design, but **rejection is an expected valid outcome**. Correctness precedes bandwidth.

## Steps

1. Require `1001_hip_internal_allreduce`; keep P2P default OFF behind an explicit runtime selector.
2. Gate on exactly two devices, bidirectional `hipDeviceCanAccessPeer`, successful peer enable, and a completed correctness probe. Capability bits alone are not activation proof.
3. Adapt every directed transfer to source-current **push** semantics. Never use a single `p2p_issuer` for both directions on gfx1100; that necessarily makes one direction destination-current.
4. Use per-direction streams/events owned by the source device, copy into destination-local scratch, then have each destination wait before the add kernel.
5. Add a startup/validation probe that transfers deterministic nonzero patterns in both directions and validates destination contents. Any mismatch disables P2P for the process and falls back to host staging.
6. Do not use kernel direct peer reads/writes in the first provider. This item is DMA/copy-engine peer transport only.
7. Preserve host staging as the mandatory fallback for unavailable/failed P2P, unsupported GPU count, or runtime errors.
8. Sweep message sizes and compare host staging versus corrected push-P2P using per-element validation on every run; raw GB/s without correctness is invalid evidence.
9. Compare end-to-end internal-AllReduce decode/prefill after the transport microbench. Provider overhead can erase a memcpy win at actual collective sizes.
10. If no stable winning envelope exists, mark the item rejected with evidence and retain it as negative knowledge.

## Detailed Solution & Technical Design

The source implementation adds peer access, one extra stream, completion events, and a P2P copy path that populates the existing `dev_tmp` scratch before the normal add kernel. BigCherry's adaptation changes ownership: each direction records the producer-compute event, sets current device to the **source**, issues the source->destination peer copy, and records a completion event whose destination compute stream waits before consuming scratch.

The provider selector must be topology/runtime scoped, not model scoped. A capability probe should include device IDs, ROCm/runtime version, both directed validations, and whether source-current semantics were used. The test must use nonzero asymmetric patterns so zero-filled/copy-skipped failures cannot accidentally pass.

No P2P path may depend on `__threadfence_system()` to make peer GPU writes visible; that mechanism has already failed on this hardware class. The copy engine and completion events define visibility for this experiment.

## Code Samples & Guidance

Pseudo-direction:

```cpp
for each src in {0,1}:
    dst = 1-src
    set_device(src)
    wait(source_compute_done[src])
    memcpy_peer_async(dst_tmp, dst, src_buf[src], src, bytes, push_stream[src])
    record(copy_done[src])
```

Then each destination waits for the transfer that targets its scratch before the local finish kernel.

## Files

- `docs/planning/active/nasone-rdna-optimizations/PNRO03.md`
- `patches/1252_nro03_allreduce_p2p_provider/{patch.toml,patch.py,SUMMARY.md,README.md,TESTING.md}`
- shared NRO static tests; future custom bidirectional hardware probe.

## Validation

Mandatory synthetic bidirectional element validation before throughput. Include repeated transfers, size edges, nonzero/asymmetric values, peer-enable-already-enabled handling, and forced fallback. Record both directed results.

Performance: microbench 1KB through largest actual collective sizes, then real model lanes. Compare host staging and P2P within the same binary/provider composition. Include GPU count !=2 as a non-selection control.

## Effort & Risk

High. ROCm may return success while data is wrong on exactly the path this optimizes. The source's fixed issuer is unsafe for known Brutus behavior. P2P can also be slower despite correct raw transfers because the CPU root complex/topology favors concurrent host staging.

## Standards

PGC02/GP13 corrected evidence is authoritative for Brutus. Never treat `hipDeviceCanAccessPeer==1` or API success as correctness. Fail closed to validated host staging.

## Acceptance Criteria

- Both directed startup probes pass bit-for-bit across repeated runs.
- Source-current push semantics are statically and dynamically evidenced.
- Any probe/runtime failure disables P2P and leaves host staging correct.
- A statistically repeatable collective-level winning envelope exists; otherwise reject rather than ship a topology guess.
- No global default before independent topology coverage.

## Notes

The source commit's P2P design is evidence/context, not an instruction to copy its single-issuer policy. This is a deliberate BigCherry adaptation prompted by real gfx1100 corruption findings.

Superseded by: PNRO03
Migration: capability-rebaseline-v3-2026-09

## Change Log

- 2026-09-08T09:50:40+10:00 (created-by): Created from nasone P2P provider plus PGC02/GP13 corrected hardware evidence; P0 correctness-first.

## Ledger-events


- Pending: ag-ledger MCP unavailable in authoring session.
- 2026-09-09T11:24:35.925666+00:00 (updated-by): Updated: section:notes
- 2026-09-09T11:42:42.936100+00:00 (state-transition): State: pending → superseded
- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:57:59.839272+00:00 (updated-by): Updated: section:ledger-events
