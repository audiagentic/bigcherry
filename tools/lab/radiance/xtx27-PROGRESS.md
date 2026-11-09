# XDP01/XDP02 progress — 2026-11-09

## Implemented and checked

- [x] Experimental worktree `the isolated XDP worktree`, branch `exp/xtx27-prefill-opt`, no changes to primary checkout.
- [x] Fail-closed dual-XTX Radiance benchmark runner using MEN03, measured binary/model/corpus provenance, dry-run default, fixed two-rank no-P2P setup, FP8 KV and drafting off.
- [x] `--embedding-placement ram` explicitly selected to preserve VRAM for long-context KV; VRAM embedding remains a separate potential short-context experiment, not a capacity default.
- [x] Existing MXFP4 build tested on both XTX. One-rep WHT6 smoke at ~704/2,305/6,679 prompt tokens: 2,085/2,581/2,534 TTFT-derived tok/s.
- [x] Two-rep matched 8192-context run at ~6,680 actual prompt tokens: exact wire, batch2048 **2,115 tok/s**; WHT6 batch2048 **2,512 tok/s**; WHT6 batch4096 **2,513 tok/s**. WHT6 has known lossy behaviour, not yet quality qualified.
- [x] Isolated per-op diagnostic: rank-0 aggregated MXFP4 `gemm_nt_q` ~2,311 ms, `all_reduce` ~821 ms, GDN chunk scan ~215 ms across prefill + 32 decode steps. Profiling forces synchronisation and is not comparable as a speed score.
- [x] Capacity-first plan XDP02, full-attention KV arithmetic with explicit layer types, TP2 prefix and single-card R9700 suffix, stage-owned state only (not implemented).
- [x] At hypothetical 1,048,576 FP8 tokens, 64 XTX TP2 layers needs 16 GiB attention KV per XTX; 40 XTX + 24 R9700 layers needs 10 GiB per XTX + 12 GiB R9700. R9700 stage weight estimates are illustrative, not runtime-resident proof.
- [x] Saved capacity envelope under `${BC_WORK_ROOT}/xtx27-prefill/1m-kv-envelope.json`; benchmark evidence under the same scratch parent, no benchmark evidence committed to Git.
- [x] 10 `unittest` card-free tests pass; Ruff check clean.
- [x] Our GPU lease released after all engines exited; no GPU processes remained.
- [x] XDP03 new BF16-based mixed-quantization plan and dry-run recipe generator, matched MXFP4 control, W4 FFN middle-prefix profile, optional R9700 suffix FP8. Local original BF16 model checkpoint/18 shards verified; no download.
- [x] Emitted matched BF16-origin baseline/mixed recipe and config-fingerprinted manifests to isolated scratch `quant-lab/` (no `.rad` conversion yet); `--write` only, default read-only.
- [x] Card-free measured-input-constrained selector tests: stage-owned logical FP8 KV reservation, per-card VRAM reserve and exact bounded memory/latency DP. Accepts `selected` generator profile only with measured evidence; no actual placement or runtime quant switching.
- [x] 31 `unittest` card-free tests pass (including 21 new XDP03 tests); Ruff clean.

## Still required

- [ ] Bench 32K-256K **native** model context with actual resident KV/GDN/graph/runtime accounting; quality, TTFT and decode measured across context and sequences.
- [ ] Exact/WHT6 correctness/KL gate and longer repeated, rotated A/B to qualify speed improvement.
- [ ] Engine `rad-convert --plan-only` under an exclusive GPU lease, then convert the BF16->MXFP4 control and mixed candidate and run KLD/rad-kbench/VRAM/perf; the RAD49 fused-gather owner currently holds the GPUs. W4 is a supported *storage* encoding, not yet proven native gfx11 IU4 instruction execution.
- [ ] Obtain comparable measured per-layer W4/MXFP4/FP8 perf, peak weights and reserves; run the XDP03 GPU-aware selector with real evidence and measure selected profile end-to-end before admitting or promoting any dynamic policy.
- [ ] Native gfx1100 MXFP4 GEMM and TP all-reduce isolated kernel experiments with correctness; identify safe speed/capacity trade-off.
- [ ] Full-layer R9700 stage + activation handoff microharness, memory planner admission and KV/GDN state ownership proof. Current Radiance helper infrastructure is MoE-only and does not implement this split.
- [ ] Future 1M checkpoint availability, true geometry, architecture and training/rope scaling support; 1M on current local Qwen3.8-27B is outside its declared 262,144-token native limit.
- [ ] Production changes only after measured gates and review, no push/PR.

Raw measurement definitions: TTFT-based tokens/s, *not* pure prefill kernel throughput. Strata Flash-Next MoE benchmarks are not matched to this dense 27B run.
