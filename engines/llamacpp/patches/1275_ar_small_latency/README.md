# 1275: small mapped-host AllReduce latency controls

## Scope

Default-off controls over the pinned mapped-host AllReduce: `BIGCHERRY_AR_SLOT_SYNC=host|none` (`none`
skips the pool-wrap host event waits for single-chunk mapped-host reductions only) and
`BIGCHERRY_AR_SMALL_BLOCKS` / `BIGCHERRY_AR_SMALL_THREADS` geometry. Defaults reproduce pristine behaviour.

## Activation

`BIGCHERRY_PATCH_HIT patch=1275_ar_small path=small_ar`.

## State and result

`evaluated` (PGC10/PGC11, 2026-10-01, 27B Q8_0 dual XTX, MTP): `slot_sync=none` neutral,
`small_blocks=1` -8.5% decode. Both arms were dropped from the adaptive-wire matrix. No experiment
contract is bound, so there is no validation adapter (it returns together with a contract).
