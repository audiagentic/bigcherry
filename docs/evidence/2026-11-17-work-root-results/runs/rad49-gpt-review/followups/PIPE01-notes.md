# PIPE01 groundwork (tail transfer/GEMM overlap), 2026-11-10 — from reading moe_remote_gpu.h and the replay numbers

How the tail layer call works today (one stream per lane, strictly serial):
1. lane_mark_rows + lane_prepare_activation_sparse: GPU kernels that PULL the routed activation rows from the engine's mapped host
   memory into VRAM (act ~1.25 ms per layer call on cold 2048-row steps).
2. route count/prefix/scatter, gate/up GEMM, gated quant, down GEMM (7.88 ms).
3. lane_route_inverse, then lane_reduce_tokens_v (one block per token row): reduces the routed contributions and WRITES the row to
   the engine's mapped host memory, publishing RAD42 chunk counters (write ~2.25 ms). It cannot start before the down GEMM of ALL experts.
All times above are per layer call (40 layers per 2048-row chunk; consistent with ~3.6-4.2k tok/s doc prefill).

Observations:
- PCIe is full duplex: the pull (host->GPU) and the write (GPU->host) use opposite directions, but the steps are serial because
  expert-major GEMM finishes every token row only at the end.
- Replay scaling of the quad kernel (old build, gate/up): 393 rows 1572 us, 1728 rows 3426 us, 2048 rows ~4090 us. Time is ~linear in rows
  (not weight-bandwidth bound), so a token-chunked GEMM (two 1024-row halves) costs roughly +20% GEMM (~+1.1 ms gate/up+down per call)
  from batch-padding and weight re-reads, versus up to ~3.5 ms of link time that two staggered halves could hide: net up to ~2 ms of
  an 11.4 ms tail call (~15% on cold steps) if the link and GEMM overlap does not contend. Unproven.
- Expert-chunked (not token-chunked) fetch is cheap in duplicated rows (the tail gets ~1.4 routes per token), but result writes cannot
  start per expert chunk without changing the summation order (bit-identity) or adding a partial-sum stage.
- Risk: the pull/write are CU kernels stalling on zero-copy reads; overlapped with the GEMM they contend for CUs. Option to test:
  hipMemcpyAsync (copy engine/SDMA) from pinned host memory for a dense or per-chunk row range, which takes no CUs. Host-memory only (no P2P).
- Cheapest experiment: a replay mode running a capture as two half-requests on two streams (token halves, staggered), measuring wall
  time per layer call against the single request; then the same with the pulls as SDMA copies.
