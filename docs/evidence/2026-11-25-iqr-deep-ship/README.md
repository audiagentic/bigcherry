# IQR 6-bit paths: three guarded native rewrites (embedding gather, head NT4, long 16-row tiles)

Three independent native rewrites of the gfx1100 plugin, each qualified separately and checked a second time by another session. Output bytes are unchanged in all three.

## A. Rotated embedding gather: wave-local butterflies
The inverse Hadamard of the rotated IQR embedding rows keeps its first five butterfly stages (spans 1 to 16) inside a wave with cross-lane exchanges and only the last two stages (spans 32, 64) go through shared memory. Workgroup barriers per gather CTA: 15 to 5; 25 to 32 fewer static instructions in each of the ten rotated forms. The unrotated forms are unchanged.
- Resource records: 17 of 17 symbols unchanged except one 4-bit rotated form, 7 to 8 VGPRs, no spill, no extra shared memory, no lost resident wave.
- Correctness: the embedding selftest passes (52 checks, zero fail, all forms, both multipliers, edge ids -1 and 512, last valid row, run boundaries); the arithmetic reproduces the original bytes (357 blocks of 128 values, identical bits).
- Replay (cold, 256 MiB flush; warm): ten changed forms median ratio 0.98 cold, 0.98 warm, all faster in paired order. A small op (about 16 us a call), kept because it is byte identical and cleaner.

## B. Vocabulary head, widest token tile (NT4): shorter weight prefetch
For the NT4 head tile only (49 to 64 rows) the next weight pair is fetched after the 256-K fold instead of overlapping it, derived from the activation staging ring size. VGPR spills 797 to 663, private bytes unchanged (204), VGPRs, shared memory and waves unchanged; the other five head symbols are byte for byte unchanged. NT1 to NT3 keep the existing path (NT3 showed a small repeatable cold regression and is excluded).
- Correctness: the head selftest and an independent two-plugin harness over five head configurations at every row count 1 to 64: zero differences.
- Replay (cold): ratios 0.90, 0.91, 0.90 at 49, 56, 64 rows (about 35 us of 360); unchanged rows read 0.97 to 1.02 (noise).

## C. Long native 16-row token tiles, chained
The accumulate-once tile (merged for NT1 and NT2) is extended to the chained NT3 and NT4 computed-trellis symbols (44 symbols) by streaming each weight fragment from the wave's staging plane at its WMMA step and keeping a single activation register plane, so the live set fits. The guard remains a function of the compile-time tile width and chain mode; the unchained NT3/NT4 and the previously excluded NT1 forms keep the old path.
- Resource records: all 3,172 kernel records compared by symbol (including all 404 16-row records): 44 symbols improve (VGPR down 48 to 68, resident waves up), none gets a spill, extra shared memory or private memory, none loses a wave.
- Correctness: hashes of the IQR row output for every case of the stock matrix plus 33/40/48/49/56/64 rows at N=512 and N=5376, all forms, chained and unchained, shipped plugin against the candidate: 0 of 602 outputs differ in either mode; the stock oracle cases pass (1,809 ok, 0 fail).
- Replay (cold, 256 MiB flush, 60 launches a point, interleaved ABBA, chained): 64 of 64 points faster (four trunk forms B6, B7, and both with bf16 scales; four trunk shapes; 33, 48, 49, 64 rows), none slower; candidate over shipped time 0.65 to 0.89, mean 0.74 (B6), 0.75 (B7), 0.81 and 0.82 (bf16-scale forms); by row count 0.77 at 33, 0.81 at 48, 0.74 at 49, 0.77 at 64. Largest on the 13312x5120 shape.

## Not included
The fused gate/up band at 17 to 32 rows (slower), the transposed-product tile (no operand layout removes the regrouping), a deeper M=1 GEMV pipeline (loses occupancy) and the head NT3 shorter prefetch (regresses) were measured and rejected.
