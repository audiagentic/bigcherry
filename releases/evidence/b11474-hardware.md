Hardware evidence for the b11402 -> b11474 bump (2026-10-08, Brutus)

Build: b-metamem-b11474b - production set (48 patches) at llama.cpp b11474 (b9acf138), targets gfx1100 / gfx1201 /
gfx1030, compiled clean. The first build attempt failed on 1347 (ggml_cuda_should_use_mmvf gained a warp_size
parameter upstream in 9d3aba6b5); fixed in the patch, no other compile error.

Smoke: Qwen3.8 Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX + R9700 tensor split, MTP drafter on the
RX 6900 XT), ctx 245760, f16 KV, ub512. Loads and generates; greedy text md5 2571b60b1505, the same as the pre-bump
production build at b11402 (run metamem-qo2). Compute arena 880.8 / 880.8 / 378.8 MiB per card, as before the bump.
Activation markers seen: 0860, 1006, 1237, 1241, 1274, 1291, 1295, 1309, 1310, 1311, 1313, 1344, 1345, 1347.

ABBA on one binary (run b11474b-ab), A = production, B = BIGCHERRY_SCHED_ASYNC_INPUTS=0 (1326 was ported into
upstream's extracted ggml_backend_sched_copy_input):

  depth  prefill A (t/s)   prefill B (t/s)   decode A (t/s)  decode B (t/s)  greedy text
  8K     1128.3 / 1169.3   1170.3 / 1169.6   82.7 / 83.4     75.8 / 76.0     identical (e383326400a4)
  24K    1173.1 / 1165.8   1167.3 / 1164.2   72.0 / 72.8     66.3 / 67.8     identical (0d8c726fbcd2)
  98K    1082.4 / 1084.7   1080.9 / 1080.7   64.6 / 65.3     60.7 / 60.8     identical (3ee76e811971)

The three greedy texts are byte-identical to the pre-bump build's at the same depths (runs qo-ab / qo-gather-ab at
b11402: e383326400a4, 0d8c726fbcd2, 3ee76e811971). Prefill matches the pre-bump figures (about 1170 / 1170 / 1084).
1326 after its port: decode +9% at 8K, +8% at 24K, +7% at 98K with identical text. The first A run at 8K (1128.3) is
the cold first load.

Probes against the CPU f32 reference (24 probes, 8K): production top-1 23/24, TV mean 0.0764, max 0.1926; production
against itself and against the 1326-off arm: top-1 24/24, TV 0.0000.

Not separately A/B'd: 1292 (k-pool tail truncate, adapted to the rewritten kpool_layout_update, no off switch). Its
effect is inside the decode figures above, which are at or above the pre-bump ones at every depth.

Outside the production set: 13 patches do not apply at b9acf138 and carry known_broken dispositions (plan item BPB01).
