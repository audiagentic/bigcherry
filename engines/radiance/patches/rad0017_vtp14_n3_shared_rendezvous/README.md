# rad0017_vtp14_n3_shared_rendezvous

## Summary

libr4d's all-reduce meets its peers through a `Rendezvous g_rv` kept in the library's own static storage. Kernel
libraries are dlopened `RTLD_LOCAL`. In a world where the XTX ranks run libr3 (BigCherry's gfx1100 build of the libr4d
source) and the R9700 rank runs libr4d, each library counts only its own ranks. No barrier ever reaches `world`, so
declare fails after the 60 s timeout. That was reproduced on cards with `--tp 2 --rank-devices`. This patch makes
every library built from the libr4d source share one rendezvous.

## Outcome

Untested (not promoted). On cards, with rad0016, mixed XTX+R9700 TP2 declares and serves Qwen3.8-27B fp8 correctly; the receipt shows the rendezvous shared by libr3.so and libr4d.so. The patch applies to v1.17.0 on top of rad0016 and is idempotent. libr3's `r3_rewrite.py --check` passes
on the patched libr4d source. `bc_shared_slot_test` covers sharing, refusal and concurrency. The mixed-card
all-reduce has not yet been run on hardware.

## Detail

### What changes

- **Core** (`core/plugin/bc_shared_slot.{h,cpp}`) adds `rad_bc_shared_slot(key, size, align, init, out, why, len)`,
  exported by the engine binary.
  - There is one zeroed, aligned slot per key for the whole process. The first caller constructs it, and it is
    never destroyed, so no callback into an unloaded library remains.
  - A key asked for again with a different size or alignment is refused with the reason.
  - `bc_shared_slot_users(key)` names the libraries that resolved a key.
- **libr4d `r4d_ar_entry.hip`** (libr3 inherits it through its rewrite):
  - `g_rv` becomes the shared slot. The key records the layout: size, alignment, rank bound, and the mutex, cv and
    link sizes.
  - A host that does not export the service (r4d_selftest, kbench, test binaries) runs a single library and keeps
    a local rendezvous.
  - A host that exports the service but refuses the slot gets a *poisoned* rendezvous: every barrier fails at
    once, with the reason. There is no fallback to the local copy, which would bring the deadlock back.
  - `rv_barrier` poisons the rendezvous and wakes every waiter on timeout, so the other ranks fail now instead of
    each waiting out its own 60 s.
- **Engine** (with `--rank-devices`): the rank receipt names the libraries that share the rendezvous.

### Limits

- Only the startup rendezvous is shared. Each library still keeps its own `ArInst`, `g_cache` and `g_ar8`, which
  hold rank-local state.
- The ranks must still agree on launch geometry (block count, variant), and N1's agreement check does not compare
  it yet.
- Graph replay and mixed-architecture coherence of pinned host memory need hardware evidence.

### Promotion record

Model: Qwen3.8-27B MXFP4 and w8a8 (.rad), tensor parallel over two RX 7900 XTX and an R9700 (TP3)
Mechanics: the patch's tests in tools/tests/patch pass and patch-lint is clean on the promotion PR; the lane builds carrying it (VTP logs named below) build and serve.
Activation: the TP3 world over (XTX, XTX, R9700) starts and serves only with this chain (rad0016 per-rank code selection, rad0017 shared rendezvous, rad0018/rad0019 uneven ownership); without rad0017 any libr3+libr4d world hangs in all-reduce init (reproduced on cards, VTP14.md), the engine reports the rendezvous as shared by libr3.so and libr4d.so and the ownership split in its startup report.
Identity: TP3 6/6/12 against an exact TP2 reference over 42,761 positions: KLD mean 0.00307, p99.9 0.084, top-1 96.80%, PPL x0.9993; the production TP2 wht6 wire scores 0.00391 / 0.111 / 96.57% on the same corpus (VTP14.md, N7, build a0979ca2). Launchers qwen27b-tp3, -int8 and -capacity start from their layouts with greedy output identical to hand-flag runs.
A/B: control cannot run (mixed-library world hangs without the chain); TP3 6/6/12 serves 64-65 tok/s at about 0.05 s TTFT, correct (VTP14.md milestone table); w8a8 TP3 KLD against BF16 within TP2's median/p99/p99.9 (VTP14.md N6).

### RUM02 slice 4 — owned rendezvous sections on Radiance 2.2

`src/bc_ar_rendezvous_impl.h` owns the process-wide rendezvous, poisoned
barrier and shared-slot lookup. `src/bc_ar_lifecycle_impl.h` owns the typed
rank allocation, published peers, deferred failure cleanup and retirement.
rad0017 materializes these files at two bounded include hooks in
`libr4d/r4d_ar_entry.hip`; hash pins bind source bytes to patch.py.
rad0021 and rad0032 attach subsequent shard and duplex edits to the owned
rank header. The three gfx11 native source rewrite files follow their
new owners. The frozen original entry source reconstructs byte-identically
when the headers are expanded; existing failed-rank retirement/concurrency
tests and native rewrite checks remain authoritative. No runtime redesign.
