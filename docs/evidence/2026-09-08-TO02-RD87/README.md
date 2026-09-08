# TO02 / RD87 hipBLASLt client smoke — Brutus GPU0

Observed 2026-09-08 UTC on `brutus`, GPU0 (`gfx1100`, RX 7900 XTX), using the
existing `hipblaslt-bench` build with the matching ROCm 7.2.4 LLVM runtime
library on `LD_LIBRARY_PATH`.

Command: `hipblaslt-bench -m 1024 -n 1024 -k 1024 --precision f16_r`.
The client reported one supported solution and completed successfully with
`RC=0`; binary SHA-256 is recorded in `binary.sha256`.

This is a client/build smoke check, not RD87's full offline signature-tuning
hypothesis evaluation.
