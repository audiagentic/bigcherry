# rad0017_vtp14_n3_shared_rendezvous

## Summary

libr4d's all-reduce meets its peers through a `Rendezvous g_rv` kept in the library's own static storage. Kernel
libraries are dlopened `RTLD_LOCAL`. In a world where the XTX ranks run libr3 (BigCherry's gfx1100 build of the libr4d
source) and the R9700 rank runs libr4d, each library counts only its own ranks. No barrier ever reaches `world`, so
declare fails after the 60 s timeout. That was reproduced on cards with `--tp 2 --rank-devices`. This patch makes
every library built from the libr4d source share one rendezvous.

## Outcome

Untested. The patch applies to v1.17.0 on top of rad0016 and is idempotent. libr3's `r3_rewrite.py --check` passes
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
