# TO02 / HI16 forced-native parity — Brutus GPU0

Observed 2026-09-08 UTC on `brutus`, GPU0 (`gfx1100`, RX 7900 XTX), with
`ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`,
`GGML_HIP_TUNE_DOUBLE_NATIVE=1`, and `HIP_LAUNCH_BLOCKING=1`.

The existing `test-backend-ops` artifact was exercised through
`python3 -m bigcherry.hi16_forced_native_parity` with screen samples 3 and
final samples 15. The four-family sweep and the MMVQ representative both
returned `PASS`; the logs assert the strict `nmse == 0` and `max_abs == 0`
parity condition. The binary SHA-256 is recorded in `binary.sha256`.

Source checkout identity for this host run: `52ab88bebc74e800645f0694327f3a8bf4965195`.
This is a bounded parity verification, not a replay-fallback or throughput claim.
