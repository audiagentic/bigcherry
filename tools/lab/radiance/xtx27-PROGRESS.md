# XDP01–XDP04 progress — 2026-11-09

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
- [x] Once RAD49 lease cleared, acquired exclusive GPU lease and ran pinned `rad-convert --plan-only` on the BF16 control and mixed W4 recipes; both PASS, 1,024 planned weights, 289 quantized/735 as-is, estimated 16.50 GiB (control) vs 16.43 GiB (mixed), no .rad files written.
- [x] Ran actual gfx1100 `libr3` shape N17408 K5120 kernel timings: at M512 W4A8I8 3226.6 µs minimum vs MXFP4 tiled 931.6 µs (W4 3.46x slower); M64 W4A16 best 368.3 vs MXFP4 tiled 308.7 µs; M1 W4A16 best 76.9 µs vs MXFP4 gated 103.4 µs / decode 146.9 µs (not matching end-to-end paths). Raw logs isolated in `quant-lab/` scratch.
- [x] STOP broad W4 prefill conversion under existing kernels (insignificant memory saving, severe kernel slowdown); runtime quality/KL not measured. GPU probes ended cleanly and exclusive lease released.

- [x] XDP04: isolated HIP gfx1100 native W4A4 IU4 WMMA microkernel + one `xtx27_w4a4.py` CLI for plan, CPU synthetic quant quality, stand-alone compile and one-leased GPU six-case suite. No Radiance patch, model conversion or runtime integration.
- [x] gfx1100 HIP compiler succeeded (warning-free); initial off-diagonal GPU mismatch traced to gfx11 interleaved accumulator/transposed W-first layout and fixed. Native W4A4 unsigned-weight/signed-activation group128 zero correction now passes all outputs of six GPU oracle geometries, K up to 1024, error zero. Lease auto-released.
- [x] 45 card-free XDP tests pass, including 14 new XDP04 cases; Ruff clean; scratch build binary/manifest and GPU logs in `${BC_WORK_ROOT}/xtx27-prefill/w4a4`.
- [x] Synthetic outlier caution: W4A4 relative L2 0.374 vs W4A8 0.193 on random 16x32x256 sampled matrices; this is not model KL. No production quality or speed claims.

## Still required

- [ ] Bench 32K-256K **native** model context with actual resident KV/GDN/graph/runtime accounting; quality, TTFT and decode measured across context and sequences.
- [ ] Exact/WHT6 correctness/KL gate and longer repeated, rotated A/B to qualify speed improvement.
- [ ] To reconsider W4: optimize XDP04's working **untiled** native gfx11 W4A4 arithmetic oracle into a genuinely fast, tiled implementation and verify it against MXFP4 including activation quantization, real-model quality and KV impact. Existing W4 broad model conversion is deferred by the 3.46x prefill-kernel regression. Do not claim the current W4 storage path uses native gfx11 IU4 WMMA.
- [ ] If native W4 becomes viable, convert BF16->MXFP4 matched control and mixed candidate and run full end-to-end KLD, TP2 throughput, memory and KV residency gates. This conversion was deliberately not started.
- [ ] Obtain comparable measured per-layer W4/MXFP4/FP8 perf, peak weights and reserves; run the XDP03 GPU-aware selector with real evidence and measure selected profile end-to-end before admitting or promoting any dynamic policy.
- [ ] Native gfx1100 MXFP4 GEMM and TP all-reduce isolated kernel experiments with correctness; identify safe speed/capacity trade-off.
- [ ] Full-layer R9700 stage + activation handoff microharness, memory planner admission and KV/GDN state ownership proof. Current Radiance helper infrastructure is MoE-only and does not implement this split.
- [ ] Future 1M checkpoint availability, true geometry, architecture and training/rope scaling support; 1M on current local Qwen3.8-27B is outside its declared 262,144-token native limit.
- [ ] Production changes only after measured gates and review, no push/PR.

Raw measurement definitions: TTFT-based tokens/s, *not* pure prefill kernel throughput. Strata Flash-Next MoE benchmarks are not matched to this dense 27B run.
