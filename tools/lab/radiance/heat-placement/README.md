# Radiance RHP01-RHP06: deterministic heat / ownership mock

**Scope: planning and simulated correctness only.** `heat_mock.mjs` contains a deliberately small, standalone CPU-only model with an executable 23-case self-test. It does **not** link to Radiance, replace kernels, exercise PCIe/ROCm or validate live migration.

Run:

```bash
node tools/lab/radiance/heat-placement/heat_mock.mjs
python3 -m unittest tools/tests/radiance/test_heat_placement_mock.py
```

Expected output has `tests: 23`, `failures: 0`. No dependencies besides Node 18+ for the mock and Python 3 for the optional unittest wrapper. Self-test returns nonzero if any assertion fails.

| Mock primitive | RHP item | What it proves / does not prove |
|---|---|---|
| `BankRing` | RHP01 | Event lag, stable order, collision/drop accounting, no future-step credit. Real Radiance ring implementation and DMA event semantics still unverified. |
| `Heat` | RHP01/RHP02 | Decay with `reports` multiplicity, phase separation, mirrored rank dedupe, reject backwards source steps and poor coverage. Split phase counts from mixed batches requires real source metadata. |
| `Planner` | RHP04 | Byte budgets, allowed-owner/protected units, per-layer max critical path rather than serial sum, shadow gain threshold and churn. Exhaustive one-move + pair-swap oracle scales only to toy sets, not 50K experts. |
| `Trigger` | RHP05 | Request-OR-time evaluation, monotonic counts and debouncing. It does not move weights. |
| `Epoch` | RHP05/RHP06 | Require all participant ACKs and zero in-flight steps before publication; abort leaves current ownership unchanged. No GPU stream fences or weight loader. |
| `makeSnapshot/readSnapshot` | RHP03 | Identity-bound deterministic canonical ownership shape and tamper detection. Mock checksum is **FNV-like and intentionally non-cryptographic**; production MUST use SHA-256 plus atomic/fsync persistence. |

## Adversarial fixtures included

- Delayed completion and nonblocking empty drain
- Ring overwrite before credit, stale completion ignored
- Out-of-order completion sorted by original step
- Same-step duplicate and negative expert counts rejected
- MTP multiplicity, prefill/decode scoring separation, mirrored TP rank deduplication
- Insufficient histogram coverage rejects owner policy
- Helper skew, protected local expert, byte overcommit, no evidence and churn penalty
- 1,000-request / 1,800-second trigger and busy suppression
- Partial ACK, inflight epoch barrier and abort preserving current map
- Snapshot checksum tamper, wrong model/cohort and duplicated expert key

## Mock contract and real implementation gap

The model intentionally simplifies the pinned Radiance source: `core/runtime/ctx.cpp` may aggregate repeated speculative passes into one GPU bank; in the mock, a bank is one `Report` with `reports` multiplicity. Exact production semantics must be confirmed from `route_report_impl`, `route_flush`, `last_histogram_reports` and `Engine::tiering_dispatch`. The mock's stage FIFO is not a proposed drop-in buffer implementation. There is no engine ABI binding, HIP transfer, 32K prefill measurement, KLD test, host GPU memory budget, end-to-end latency, or hardware evidence in this package.

The feature must remain experimental until the per-plan GPU/parity/quality acceptance gates pass. See `docs/planning/active/patching-radiance-heat-placement/README.md` and `docs/evidence/RHP01/mock-results.json`.
