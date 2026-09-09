# RHA04 current BC-native smoke disposition

Date: 2026-09-10  
Execution environment: Brutus role resolved through environment settings; no
host connection details are committed.

This is an invalid/incomplete smoke attempt, retained so it cannot be reused
as performance evidence. The cached BC-native 27B dual-XTX server binary was
launched with diagnostics enabled, `HIP_VISIBLE_DEVICES=0,1`,
`ROCR_VISIBLE_DEVICES=0,1`, tensor split, and the production model. The
process never reached `/health` or emitted `model loaded`/`listening`; it was
terminated after the bounded observation window. No benchmark request ran.

The log ended during model materialization after reporting a virtual `Meta()`
device and layer assignment to `Meta()`. This is not evidence that the binary
cannot execute on the GPUs; it is an infrastructure/model-load timeout and is
not comparable to the prior completed server-bench cells.

Remote raw-log hashes:

| Attempt | SHA-256 |
| --- | --- |
| detached observation | `5a5f46579e0cbbea1dfa4936f2250d629ff78fd3a34847eb84dd1f17baa9e38a` |
| bounded SSH observation | `c415c2afe19e52c2047e142091f833efa71d65b619fcf7797fd1a29d1439f1ec` |

No throughput, correctness, RCCL, or 0840 conclusion is drawn. A fresh
decision-grade run must first establish a known-good binary/model materialize
and health lifecycle, then use the required physical-device evidence contract
before timing.
