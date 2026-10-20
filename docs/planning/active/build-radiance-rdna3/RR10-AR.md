## RR10-B: gfx1100 all-reduce phase map (2026-10-12)

Qwen3.8-27B hidden width is 5120. Exact is bf16 (2 B/element). WHT6 is 6-bit codes
plus one bf16 scale per 64 elements: 0.75 + 2/64 = 25/32 B/element, 25/64 of exact.

| rows | elements | exact | wht6 |
|---:|---:|---:|---:|
| 2048 | 10,485,760 | 20,971,520 B = 20 MiB | 8,192,000 B = 7.8125 MiB |
| 64 | 327,680 | 655,360 B = 640 KiB | 256,000 B = 250 KiB |

The dense trunk has a mixer-output reduction and an MLP-output reduction per layer: two
logical reductions per layer. A scheduler feeding 64-row quanta performs both for every
layer once per quantum; a 2048-row chunk contains 32 such quanta. The embedding reduction
is a separate exact-only boundary and is not counted in the 2/layer.

Device/host ownership to verify against the lab source dump:
- device: WHT transform/quantise and fused reduce + residual/RMSNorm/next FP8 quantiser;
- no-P2P relay: device->pinned-host, peer rendezvous/fold/wake, pinned-host->device;
- device: payload decode/reduce and fused norm/quant output.

`ar-phase-profile.sh` records HIP runtime, kernel-dispatch and copy-engine events for exact
and WHT6. `ar-source-map.sh` prints the exact pinned-host implementation from the lab checkout
so a D2H->H2D gap is not guessed to be CPU fold time.

Candidate decisions before code:
1. Larger/fewer prefill messages: STOP. Existing 1024/2048/4096 chunk runs are flat within
   run spread, so this lever already failed its serve gate.
2. Fuse unpack with next quantise: verify before changing. `ar_rmsnorm_quant_fp8` is already
   the fused consumer on the device path; do not duplicate a fusion that already exists.
3. Rank-A send overlapped with rank-B pack: implement only if source map shows serialization
   in libr3/plugin-owned code. If serialization is in the core host relay it is outside this
   task's allowed source ownership.
4. Prefill-only row-scaled 8-bit wire: non-bit-exact. Implement only if WHT6 pack/unpack
   dominates enough to repay the larger payload; gate with `kld-tp-wire.sh`. Exact and WHT6
   remain unchanged when the experiment is disabled.

Stop B optimization if no removable libr3 stage is about 30 us/call or more, or if D2H+H2D
plus unavoidable host rendezvous accounts for most of the ~145 us call. Then the no-P2P
transport floor, not the gfx11 codec, owns this item.
