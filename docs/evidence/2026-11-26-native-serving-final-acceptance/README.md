# Native serving cutover: final acceptance (installed release 2020.23.0)

The cutover (RSC01 to RSC06, RSC11, RSC14, PWR08, PWR10) is accepted on the installed release, not on an experiment build. One run per check, one lease window each. Radiance pin 2.2.0, unchanged.

| Check | Release | Result |
|---|---|---|
| Install, verify, start, canary and stop of both layouts (Gate E) | 2020.19.0 to 2020.23.0 | pass (2020.23.0: 230 s and 21 s to running) |
| Mixed-architecture dense TP3 and helper/MoE start on stock `radiance --config` with a sealed admission: start, no `default` alias, SIGHUP refused, admin reload and unload 403, stop order with power restore, stored-plan reuse, engine SIGKILL ends the owner with no restart | experiment build of the cutover branch, then every release since | pass (see the RSC02-B acceptance page) |
| Minimum-cap apply, readback while serving, exact restore, receipt-qualified second apply, unqualified cap refused (PWR10) | 2020.21.1 | pass |
| Host-memory contention: named early refusal (exit 97), stored plan kept, memory released, reuse start (RSC14) | 2020.22.0 | pass |
| Quality through the native offline engine: engine on `--config` plus the admission, no `--kld` flag, KL result and per-row output, nothing left running | 2020.23.0 | pass: KLD mean 0.00314, p99 0.0134, top-1 0.967 over 42,761 positions (a different corpus from the model's recorded 8,240-position figures, so not comparable with them) |
| Flash-Next parity, one launch against the stored baseline | 2020.23.0 | pass, all nine gated metrics within tolerance |

Flash-Next parity detail (baseline value, this run, change; a gated metric fails only when worse than its tolerance):

| Metric | Baseline | This run | Change |
|---|---|---|---|
| decode, 1 stream, ms per step | 24.43 | 24.50 | +0.3% |
| decode, 6 streams, ms per step | 47.22 | 48.67 | +3.1% (tolerance 5.2%) |
| decode, long context, ms per step | 26.35 | 26.51 | +0.6% |
| decode, 8 sessions, ms per step | 28.58 | 29.20 | +2.2% (tolerance 3.3%) |
| document prefill, tokens per second | 3678 | 3725 | +1.3% |
| long prefill, 7 and 8 | 2963, 3067 | 3020, 3126 | +1.9% |
| cold first token, s | 8.30 | 8.26 | -0.5% |
| 8-session first token, s | 22.01 | 21.84 | -0.8% |

The comparison command refuses a single launch (it needs two distinct launches), so the values were computed from the launch record against the baseline file with the harness' own metric extraction and the stored tolerances.

Defects the acceptance found and fixed: the native Fleet admitted only cards of the first card's architecture (mixed TP refused); the owner canary still sent the removed `default` model name; the engine plan check reported a host-pressure mismatch as drift (RSC14); a KL run kept the authored speculative depth (now scored with depth 0); the power-apply path had no end-to-end test. Open and unexplained: the first Gate E run of 2020.21.0 failed in both layouts and its logs were not kept (a failed run now keeps them); the same archive passed on rerun.

Two leftover shared-memory header files (8 KiB each) from starts that ended without retiring their segment predate these runs; none of the runs here left one.
