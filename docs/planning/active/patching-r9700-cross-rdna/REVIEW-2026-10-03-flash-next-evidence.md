# R9X review against current Flash-Next work (2026-10-03, claude)

Scope: how tonight's measured Flash-Next results (Brutus: 2x gfx1100 + gfx1201 tensor split, gfx1030 MTP
draft, no P2P, stock kernel) change R9X priorities. Evidence lives in `tools/lab/flash-next/` and the
Brutus run directories named below.

## Planning-tool issue (blocks normal lifecycle)

The R9X files have no planning frontmatter and the `R9X` prefix contains a digit, so the ag-planning tools
reject every ID (`VAL-PLN-035 invalid planning item ID`): items cannot be listed, linked to ledger events,
reviewed or state-changed. Recreate them through `plan_create_item` with a letters-only prefix (for example
`RNX01`..`RNX11`, or `RXS`) before implementation starts, keeping the R9X numbers as aliases in titles.

## R9X01 — baseline evidence already captured

Config: `-ts 2,2,3`, 192K, q8_0 KV, ub512, MTP3 draft on gfx1030, `--allreduce cpu-root`.

- Prefill, 47K prompt (rocprof, `flashnext-long-ctx-profile-1`): RCCL 30.3% of summed kernel time
  (8832 AllReduces per rank, ~1.34 ms each at 3.3 MB); flash_attn_ext_f16 8%; MoE/quant matmuls ~25%;
  f32 rocBLAS GEMMs ~11% (17472 calls = 48 layers x 2 HC modules x down+up per ubatch, i.e. the R9X04 HC
  target); dsv4_hc_post 2.4%. Same-GPU D2D copies 7.6 s (276 per GPU, bimodal 0.35 ms / 20-40 ms spikes on
  gfx1201, source not yet attributed); H2D 4.7 s (~23 per token, PLE gathers: R9X08).
- Decode at ~80K cached (`flashnext-long-ctx-decode-3`, decode window only): GPUs 20-24% busy (draft GPU
  12%), cpu-root consume spin 5-11%, flash-attn + q8_0 KV dequant 1.2-1.8 ms/token per GPU, QSA top-k +
  bf16 mul_mat_f ~1 ms/token total. R9X02's QSA path is minor at this depth. rocprof overhead is large:
  unprofiled decode at 80K is 38 t/s vs 27.8 profiled, so use unprofiled timings for E2E claims.
- Host side (`flashnext-long-ctx-perf-1`): an O(n_ctx) kpool layout rebuild ran every MTP step. Fixed by
  `1292_kpool_tail_truncate`: -14 to -15% ms/step at 80K, -2 to -3% at 10K (two ABBAs,
  `flashnext-kpool-ab-1/2`). After 1292 no libllama symbol exceeds ~0.5% (KQ mask ~0.4%); the remaining
  80K vs 10K gap (~66 vs ~51 ms/step) is GPU-side.
- The meta backend splits each rank graph at every PARTIAL (96 subgraphs per step), which supports the
  README's launch-fusion-first ordering (R9X03/R9X04/R9X10).

## R9X07 — the proposed carrier lost in-model

The host round trip that R9X07 proposes to carry WHT over was measured with f32 payloads and loses to RCCL
at every chunk size (Flash-Next `-ts 4,4,3` ub1024 prefill, auto 1447-1452 t/s): 1 MiB 1057, 2 MiB 1057,
4 MiB ~1027, 8 MiB 937 t/s (`flashnext-chunk-sweep-1`). Four sum workers changed nothing (run 6), so neither
the memop count nor the CPU sum is the cost; the in-model cost per 6.5 MB AllReduce (~6.3 ms) is about twice
the isolated microbench (2.9 ms). The 1291 large path is opt-in, default off (d16f1dd6).

- WHT4/6 must therefore win through byte reduction alone. Measure in-model from step 1; microbench numbers
  for this path did not transfer.
- The RCCL baseline to beat is `NCCL_ALGO=Tree NCCL_PROTO=Simple`: +2.0% prefill, ABBA with complete
  separation (`flashnext-ctx-fit-2`). CTA counts and Ring/LL128 were flat (`flashnext-rccl-algo-sweep-1`).
- Small decode messages: the cpu-root 64 KB threshold is near optimal (16K-256K sweep, 128K about +1%,
  within noise).

## R9X11 — audit input for 1291

- Small path: single-block produce/consume kernels, generation derived on device (graph-replay safe);
  uncomputed meta ranks contribute zeros.
- Large path (opt-in, default off): generation `l_gen` is a host counter incremented at enqueue and baked
  into `hipStreamWriteValue32`/`hipStreamWaitValue32` arguments and the descriptor ring, so it is NOT
  graph-capture safe. It is correct today only because meta-backend AllReduce calls run outside HIP graphs.
  Per-chunk flags are monotonic generations compared with `>=`, so mixed message sizes cannot create parity
  divergence. Record it as "not graph-safe, off by default", or remove it if R9X07 does not adopt it.

## R9X06 — overlaps the MET group

R9X06 does not reference `patching-moe-expert-tiering` (MET01-MET06: cold experts on the 6900,
routing-driven placement, MET02 range MUL_MAT_ID). Both consume 1279 routing dumps and both change where
expert weights live. Decide ownership first: MET owns static placement (the static-frequency baseline R9X06
already requires) and R9X06 owns dynamic LRU only, or merge them. Context is VRAM/compute-buffer bound at
192K (every rebalanced `-ts` ran out of memory, and q4 KV is ruled out by the owner), so expert offload is
one of the few remaining context levers.

## Other notes

- R9X02: at 80K decode, attention costs ~1.2-1.8 ms/token per GPU, including ~1 ms/token of q8_0->f16 KV
  dequant on the MTP-verify tile path. The verify-width path is a better target than QSA scoring here.
- R9X08: check whether PLE rows are sent to all three tensor-split ranks (GPT suspicion); ~23 H2D per token.
- Patch slots: 1292 is now consumed by `kpool_tail_truncate`, which the README already reflects.
