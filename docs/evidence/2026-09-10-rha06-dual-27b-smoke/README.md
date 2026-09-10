# RHA06 declarative runtime-matrix dual-XTX 27B smoke

Date: 2026-09-10
Plan item: RHA06
Evidence role: wiring/liveness smoke, not performance admission or RHA04 attestation

The pushed `bigcherry runtime-matrix` entrypoint resolved one registered
27B Q8_0 model, the configured dual-XTX physical topology and the registered
`production-dual-xtx` runtime profile, then delegated to the maintained
`bench/run_bench.py --bench-type server-bench` runner.

The native server loaded on the two configured XTX devices and the maintained
`tg128` liveness configuration completed with `tg128_tps: 34.24`. This is a
single-repetition wiring smoke only. It is not a parity result, a promotion
decision, or proof of ordered physical-device attestation for RHA04.

Observed invariants:

- matrix state: `completed`, one dual-device cell executed;
- canonical visibility recorded as `ROCR_VISIBLE_DEVICES=0,1` and
  `HIP_VISIBLE_DEVICES=0,1`;
- `resolved-matrix.json`, atomic `status.json`, and append-only
  `events.jsonl` were emitted;
- the server was stopped cleanly after the smoke;
- no performance admission was attempted.
