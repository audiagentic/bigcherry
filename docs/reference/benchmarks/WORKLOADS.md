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
bigcherry workload compare prefill-doc-44k
bigcherry workload compare pwr-phases --layout flashnext-2xtx --since 2026-10-27
```

`run` requires a **running** OpenAI-compatible server; it sends inference traffic, so a lead must first arrange GPU authorisation. The CLI **does not start/stop processes, touch GPU power settings, or take leases**. Use `--manifest` when the active engine's exact manifest cannot be verified via its process command line. Never guess a build, layout, policy state or calibration key from whichever historical manifest happens to be newest.

Each completed run produces exactly **one** JSONL entry in `$BC_WORK_ROOT/results/index.jsonl` (outside Git) and a saved full record in `$BC_WORK_ROOT/runs/workloads/<date-id-uuid>/result.json`. The row includes UTC timestamp, `workload_id` and `workload_version`, median and per-rep metrics, measured prompt counts, build path/build ID and source commit **if evidenced**, layout, kernel, policy state and calibration key if available, manifest options (spec depth, chunk, host pool/KV, sessions), per-BDF sysfs readback (`power1_cap`, VDDGFX offset, OD SCLK max, forced perf level), and run folder. Unknown historical metadata stays unknown, not “stock”. Power state is an observed snapshot, not an instruction to set it. A changed power snapshot during a run is marked.

`compare` restricts rows to the **same ID and version**, optionally layout and date, and prints differences in build, layout, kernel, policy, power, manifest options, corpus, phase duration/sensor and repetition count. Comparisons are **not** proven like-for-like merely because the IDs match; differences and unknown metadata require a caveat or repeat under matched conditions.

## Historical index

`bigcherry workload seed` imports identifiable completed artifacts idempotently; seeded rows have `source=seeded` and evidence paths, with no invented run-level sysfs readback. Import sources:

- `runs/policy-onoff-ab`: `on1/off1/off2/on2` document slices, cold27k, decode 1-stream and 6-stream. Expected 16 rows, subject to artifact/manifest validation.
- `runs/pwr-stock/docpf.out`: six 44k document slices with policy OFF, retained explicitly as **6/10 partial** (so it cannot silently be treated as the ten-rep baseline).
- `runs/pwr01-baseline-10271708`: October 27 **valid preserved** PWR01. The October research page's older `10271712` label was corrected to `10271708`; `10271703` was prefix-cached and is excluded.
- `runs/pwr01-baseline-11111036` (4GPU stock), `11111051` (4GPU -125mV reported), `11111104` (2XTX stock). The latter corresponds to `runs/pwr-stock/out2xtx.log`. Historical power labels are descriptive, not a substitute for an in-phase sysfs readback.

The seed can be rerun without duplicating identified artifacts; outputs live only in the outside-Git index. Historical PWR01 raw response-level token timing and policy A/B corpus SHA-256 were not retained. These are **historical evidence**, not validated fresh reproductions.

**Evidence rule:** Every performance claim in a PR, issue, plan or benchmark report must cite `workload-id@version` **and a concrete `row_id`** from `$BC_WORK_ROOT/results/index.jsonl` (or a saved result JSON). State any comparison warnings and exclude unlike measurements from an attributed improvement claim.
