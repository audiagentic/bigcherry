# Canonical serving workloads and results

The **one registry** is [tools/bigcherry/bench/workloads.toml](../../../tools/bigcherry/bench/workloads.toml). Each workload has an ID and a monotonically increasing version; change the version when the stimulus, scheduling, metric definition, or repetition policy changes.

| ID @ version | What it measures | Protocol |
| --- | --- | --- |
| `prefill-doc-44k@1` | **Prefill**, prompt tokens / request wall time | `/v1/completions`, ten sequential slices 1–10 of repository `docs/**/*.md`, original `doc_prefill.py` offset and 3.9 chars/token rule, target 44k (~44–50k actual), 1 output token, `temperature=0`, `ignore_eos=true`. Corpus SHA-256 retained, and checked on comparison. |
| `cold-27k-ttft@1` | **Cold TTFT**, milliseconds to first stream output | Eight individual 1-stream `stream-bench` runs, calibrated 27k prompt tokens, seeds 501–508, eight generated tokens each. |
| `decode-1stream@1` | **Decode** finite-batch aggregate tokens/s | Existing `stream-bench`, 1 stream, calibrated 900-token prompt, seed 7, 256 output tokens, 2 reps, `temperature=0`; engine decode rate also retained when available. |
| `decode-6stream@1` | **Decode** finite-batch aggregate tokens/s | Same stream-bench inputs, 6 simultaneous streams, 2 reps. |
| `pwr-phases@1` | **Separate decode and prefill** tokens/s and board watts | PWR01 phase order idle-loaded / lighthouse 1024-token decode / unique-prefix 800× fox-sentence 1-token prefill / idle-after, 60 s each plus idle settling. Today's `run` is against an already-running server (no launcher/lease/power writes). Historical PWR01 used 40 s or 60 s and AMD-SMI; the new client samples read-only sysfs hwmon; the index flags phase/sensor differences. |

**Never call an unqualified number “prefill”.** The document-slice and PWR01 fixed-prompt rates are different workloads. Decode is separately named, never combined with prefill. Stream-bench aggregate rate is finite-batch makespan throughput, **not** instantaneous token generation rate. Cold TTFT is a latency, not prefill throughput.

## Commands

From a checkout with `PYTHONPATH=tools` (or installed `bigcherry`), first load the host environment so `BC_WORK_ROOT` resolves to the shared results directory:

```bash
source tools/env/bigcherry-env.sh
bigcherry workload list
bigcherry workload run prefill-doc-44k --url http://127.0.0.1:18452
bigcherry workload run decode-6stream --url http://127.0.0.1:18452 --manifest $BC_WORK_ROOT/flashnext/manifest-EXACT.json
bigcherry results list --engine radiance --workload prefill-doc-44k
bigcherry results history --engine radiance --workload prefill-doc-44k --metric prefill_tps
```

`run` requires a **running** OpenAI-compatible server; it sends inference traffic, so a lead must first arrange GPU authorisation. The CLI **does not start/stop processes, touch GPU power settings, or take leases**. Use `--manifest` when the active engine's exact manifest cannot be verified via its process command line. Never guess a build, layout, policy state or calibration key from whichever historical manifest happens to be newest.

Each completed run publishes exactly **one** immutable run record (run-record v1, see `docs/reference/tooling/TOOLING.md`) and keeps its run folder, with the workload's own result row, under `$BC_WORK_ROOT/results/workloads/<date-id-uuid>/`. Compare runs with `bigcherry results compare`, `baseline` and `history`, which read only SHA-verified records; they report differences in build, layout, kernel, policy, power, options, corpus and repetition count, and comparisons are **not** proven like-for-like unless that report says so. There is no results index file and no importer of older logs: the 138 rows of the 2026-11-11/12 runs, whose manifests were purged, are summarised in `docs/evidence/2026-11-20-retired-workload-index/`.

**Evidence rule:** Every performance claim in a PR, issue, plan or benchmark report must cite `workload-id@version` **and a concrete `row_id`** (its run-record SHA-256 from `bigcherry results list`). State any comparison warnings and exclude unlike measurements from an attributed improvement claim.
