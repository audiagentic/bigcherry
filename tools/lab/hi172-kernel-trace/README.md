# HI172 kernel-level trace: does the tuned replay path actually help in real serving?

Plan item: HI172 (see also HI171, HI132, HI141, HI143)
Status: active
Owner: HI171/HI172 investigation
Question state: open

## Question

HI171's real e2e matrix found isolated per-dispatch tuner speedups (5.7%-35%
across five caches) do not reliably translate into real end-to-end
performance on any topology, and GPU0 shows a real regression. HI172 formalized
a taxonomy for why (dilution / off-critical-path / in-context regression /
system-neighbor externality / an untried slack-exploiting inverse), reviewed
by dev-gpt-agent (req_714d201cc4b64f87). A first single-shot rocprofv3 trace
on GPU0 was inconclusive (native vs replay total GPU kernel time differed by
only +0.01% in one sample) -- GPU0's real regression (~1.3-1.8%) is small
enough that a single trace can't separate it from noise.

This experiment builds a REPEATABLE, statistically honest version of that
trace: N profiled passes with interleaved unprofiled controls (HI132's own
environment-drift-detection discipline), aggregated per-exact-kernel-symbol,
targeting the dual-XTX 27B topology specifically (the most common real
deployment shape) rather than GPU0.

## Inputs

- Real production native binary (BC dispatch, no replay) and replay binary +
  the current-source-fresh dispatch.cache from HI171's 27B re-export
  (hi171-27b-dual-reexport-20260908), both already built this session.
- HI132's own primitives: `bigcherry.profiling.rocprof.rocprofv3_command_prefix`
  / `parse_kernel_trace`, `bigcherry.tuning.server_runner.ServerRunner`
  (command_prefix hook).
- A long (~2048+ token) prompt, matching the scale of HI171's pp2048 e2e
  measurement (the metric that showed the clearest divergence pattern across
  the matrix).

## Outputs

Generated outputs go under `artifacts/lab/hi172-kernel-trace/`: per-pass
rocprofv3 CSVs (kernel/HIP-API/HSA-API/memory traces), a per-pass unprofiled
control wall-clock measurement, and an aggregated `summary.json` with
per-exact-kernel-symbol calls/mean/p95/total for native vs replay across all
passes plus the interleaved control spread (environment_stable flag, same
semantics as HI132's own >5% drift threshold).

## Runtime

GPU required: yes (dual RX 7900 XTX, devices 0,1)
Real compilation required: no (reuses already-built production binaries)
Mutates canonical BigCherry state: no (reads existing caches/binaries only)

## Safety

- Canonical-state mutation: none -- read-only against existing builds/caches.
- Do not import this experiment from `bigcherry` production, tests, or
  maintained analysis.
- Stop conditions: abort and report if either arm's server fails health/
  shutdown checks, or if `parse_kernel_trace` finds an empty/malformed CSV.
- Cleanup: `run.py` always shuts each ServerRunner down via its own context
  manager (SIGINT/HTTP `/shutdown`) even on failure; verify no orphan
  `llama-server` process remains after a run (`ps aux | grep llama-server`).

## Disposition

Not yet complete. When the question is answered (a statistically-supported
verdict on which HI172 taxonomy bucket explains the divergence for 27B
dual-XTX), record the result here and either delete this experiment or
graduate the repeatable-trace capability into `bigcherry.profiling` /
`bigcherry profile-campaign` (HI132) proper, per gpt's original
extend-don't-rebuild instruction.
