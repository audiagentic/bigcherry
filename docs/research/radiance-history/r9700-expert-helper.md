# The R9700 as an expert helper: what was tried and what came of it

History page (2026-10-20). The code of these attempts is superseded and was not kept; this page keeps what was tried, what was
measured and why each step was taken or dropped. Current state: [../../reference/engines/radiance/models/](../../reference/engines/radiance/models/)
and plan items RAD06, RAD13, RAD14, RAD15.

## 1. XTX-primary, R9700 as expert helper (RAD03, Oct 2026, PRs #231 and #233, not merged)

- Layout: Flash-Next served on the two RX 7900 XTX (tp 2, libr3); the R9700 (gfx1201) ran a separate helper process that owned
  cold routed experts in its 32 GB and computed them with libr4d's own grouped GEMM, through a pinned-host mailbox with GPU-side
  flags, one R9700 stream per XTX rank, runtime slot maps from the engine. Engine side: patch rad0002 (publish, rescatter,
  post-gather add) plus shadow, mask-only and poison diagnostics (now rad0004).
- Ownership was budgeted from the container's expert profile (slice 2b), later from a measured heat histogram.
- Quality (4-document KL): helper-off, shadow and mask-only exact; helper-on KLD 0.062, top-1 88.8%. The gap was never fully
  explained at the time; PR #233 was held for an owner decision ("merge" vs "keep digging").
- Why it was dropped: the R9700 sits behind a PCIe 4.0 x4 upstream link (the engine's own startup report names it), so every
  routed layer's activations and results crossed the narrowest link in the box (the kernels README of the time measured 3-5 GB/s,
  ~1.0-1.7 ms per traversal of a 5 MiB prefill message); and the owner's direction (2026-10-13) that the R9700 is compute, not
  storage, led to the inverted layout below.
- What carried forward: the mailbox protocol, slot maps, ownership by heat and the rad0002 hand-off are the same design in today's
  moe-remote worker (engines/radiance/components/moe-remote), which builds for gfx1100 and gfx1201.

## 2. R9700-primary, XTX as expert workers (RAD06 onward, current production layout)

- The R9700 runs the engine with libr4d (native gfx12 kernels); the two XTX each own about 20 GiB of cold experts. With them the
  engine's host-resident weights fall from 14 GiB to 77 MiB a rank, so prefill no longer streams experts over the x4 link.
- Flash-Next decode (MTP 3, 2026-10-20): 115-120 tok/s at 1 stream, 201 at 2, 261-276 at 4, 365 at 6. The XTX tp2 engine alone
  (no helper) measured 45-65 / 119-137 in the same week.
- Decode kernels on the workers were rewritten in RAD14 (FP8: 451 -> 141 us a 16-row 35B request; W4: 101 -> 77 us a 4-row
  Flash-Next request).

## 3. Is the R9700 helper worth revisiting?

The capability exists on main: the moe-remote worker runs on gfx1201 (35B FP8 decode request on the R9700: 115 us with our
packed-f16 GEMV, 135 us with libr4d's grouped GEMM). An XTX-primary layout would now be worth one A/B only once the XTX engine has
the fast decode rows (RAD15 step 6); the x4 link remains the main argument against it.

## Other unmerged work summarised here

- PR #86 (`feat/rad11-gfx1100-bootstrap`): an out-of-tree `libr11` gfx1100 plugin built from scratch (BF16 add/mul, a 59-kernel
  libr4d audit, a topology probe). Superseded by libr3, which adapts libr4d's own sources instead of rewriting them. Its plan group
  (build-radiance-rdna3, RR01-RR10) is kept as superseded, the audit as `docs/research/radiance-libr4d-gfx1100-audit.md`, and its
  notes as `R3_MOE_HELPER-2026-10-12.md` and `kernels-README-2026-10-12.md` here.
