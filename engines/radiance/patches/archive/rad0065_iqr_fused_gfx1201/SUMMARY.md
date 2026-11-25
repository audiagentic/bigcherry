# rad0065 — native-WMMA single-band IQR prefill

**Status:** validated
**Plan item:** none

## Summary

Use block-local decoding rather than publishing a global integer fragment plane for the single-band prefill tile on gfx1201. Selection is a constexpr certificate for computed trellis kind 6, including the qualified 16-level row-f32 and bf16 forms and the 15-level row-f32 form; only one activation band may use this route.

## Outcome

Code only, untested. Engine build, byte-identity, resource inspection and replay gates remain for the validating session. The fused kernel must have no VGPR spills, and the block's LDS allocation must fit the target device. The existing plane route is unchanged for every other form and for multiple activation bands.

## Detail

A narrow, fail-closed two-file transform adds one block-local weight-fragment producer and a constexpr dispatch guard. Each of the eight waves decodes one 16-row unit into the native fragment layout and publishes its two 64-K halves in local storage. The existing prefill body still executes the original integer WMMA K order, scale fold and BF16 store. No gfx1100 fragment shim, global decoded weight-plane write, producer/consumer atomics or spin protocol is introduced. The package includes application, idempotence, negative-anchor and routing tests.
