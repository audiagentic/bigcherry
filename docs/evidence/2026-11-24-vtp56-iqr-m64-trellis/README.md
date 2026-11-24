# IQR wide-tile trellis resource audit

## Stage 0 — scope and dispatch

The existing 16-row native-accumulation change is present. The queued native-fragment change is not part of this branch's base. The kernel under audit is the narrow tile for 17–64 tokens. Larger token counts dispatch a separate prefill tile: M=65–256 comparisons alone cannot qualify this rewrite.

## Stage 1a — baseline metadata

The gfx1100 wide tile has 544 compiled specializations, all with register spills. There are 464 trellis specializations: 96 table-trellis (kind 5), 368 computed-trellis (kind 6). All 544 use 240–256 VGPR and six register-file waves per SIMD. The largest private allocation is 3,640 bytes per thread. The separate report records all 544 baseline VGPR/spilled-VGPR/private-byte/wave records.

This is a baseline only; no candidate resource gate, integer-sum identity, byte-identity, full build, IQR selftest, GPU timing replay, serving test or acceptance claim is implied.
