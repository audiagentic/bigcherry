# RHA06 declarative runtime-matrix maintained-harness smoke

Date: 2026-09-10
Plan item: RHA06
Evidence role: wiring/liveness smoke, not performance admission

This run exercised the pushed `bigcherry runtime-matrix` entrypoint on the
configured build-server role. The matrix resolved the registered 9B model,
one physical GPU, the registered `production-safe-single` runtime profile and
the native build identity before launching the maintained
`bench/run_bench.py --bench-type server-bench` delegate.

The server was a production-shaped native llama-server launched with the
environment-provided device visibility. The maintained harness completed its
`tg128` liveness configuration and returned `tg128_tps: 85.4`. This value is
not a performance claim: it is one repetition used to prove the configured
run boundary and teardown path.

Observed invariants:

- matrix state: `completed`, one cell executed;
- canonical visibility recorded as `ROCR_VISIBLE_DEVICES=0` and
  `HIP_VISIBLE_DEVICES=0`;
- `resolved-matrix.json`, atomic `status.json`, and append-only
  `events.jsonl`, and atomic `summary.json` were emitted;
- the server was stopped cleanly after the smoke;
- no production admission or parity conclusion is inferred from this run.

The JSON artifacts in this directory are copied directly from the run output;
the model, binary and runner paths remain identity evidence, while host
connection details continue to come from the environment configuration.
