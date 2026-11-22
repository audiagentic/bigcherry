# LTX03: native three-rank text cross-attention declaration

**2026-11-22: host-only PASS. Production LTX TP3 remains refused.**

## Scope

The existing rad0060 LTX attention builder and C02 model-backed native harness now declare text cross-attention for all 48 DiT blocks, both video and audio, both legal 8/8/16 and 12/12/8 head splits, and all three ranks: **576 native text cross-attention declarations**. The prior all-block self-attention test covers 576 additional declarations.

Q uses a one-bank IQ8R row slice and independent KV uses two row-sliced banks; both use the same native weight, bias and layout handling, global exact Q/K RMS operations and existing deferred output reduction with one BF16 bias. The test exercises four query rows against two KV rows and refuses an unquantized KV input before mutating weights. No new graph, collective, loader or kernel family is added.

## Failing-first evidence and verification

1. Original descriptor refused bank count 1; after admitting it, native per-part spans refused one non-stacked row bank. The fix reuses the native single-span API for Q, Q bias and scale and retains the existing multi-part API for two-bank KV.
2. Source-managed rad0060 applied **23 edits** against pinned Radiance 2.2 and is **idempotent**. Its generated C++ and header are byte-identical to the compiled isolated stage.
3. Nice-19, single-worker host C01/C02/C05 build: **PASS**.
4. Native CTest: **12/12 PASS**, including 576 real-model cross-attention declarations, 576 previously supported self-attention declarations, C06 issued host checkpoint attention, and the host-only AQL refusal gate.
5. Patch unit pytest: **3 passed, 1 skipped** (optional environment-dependent real-source pytest); the real-source patch replay was run separately and passed.

No GPU ran in this slice. Declaration and host C06 execution are not proof of three-rank text cross-attention issue or TP1 numerical parity.

## Remaining ownership

LTX03 next: native cross-modal a2v/v2a with unequal channel widths, FFN/residual/modulation, actual GPU issuance of all attention kinds and complete 48-block DiT. LTX02 owns captured TP1 activation parity and 1/16/64 rows. RUM14 owns the production Engine-resident loader. LTX04 owns native three-rank serving and a real generated clip. TP1 behavior and the production world-size refusal remain unchanged.
