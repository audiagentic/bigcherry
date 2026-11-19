# Canonical Radiance serving workloads and immutable results

The workload registry is [tools/bigcherry/bench/workloads.toml](../../../tools/bigcherry/bench/workloads.toml).
Change the version whenever the stimulus, scheduling or metric contract changes.

| ID @ version | Measurement |
| --- | --- |
| `prefill-doc-44k@1` | Document-slice prefill throughput; original offsets and corpus SHA-256 preserved |
| `cold-27k-ttft@1` | Cold first-token latency for eight independent seeds |
| `decode-1stream@1` | Finite-batch decode aggregate tokens/s, two measured repetitions |
| `decode-6stream@1` | Six simultaneous decode streams and separate engine metrics |
| `pwr-phases@1` | Decode, prefill and observed board power in separate phases |

## Commands

Source the environment to select the effective work root, then use the
installed CLI. Running a workload sends traffic to an existing owner and
requires the operator's GPU authorisation. It does not start a server, apply
power settings or acquire a lease.

```sh
source tools/env/bigcherry-env.sh
bigcherry workload list
bigcherry workload run prefill-doc-44k --url http://127.0.0.1:18452
bigcherry workload run decode-6stream --url http://127.0.0.1:18452 --manifest "$BC_WORK_ROOT/flashnext/manifest-EXACT.json"
bigcherry results list --engine radiance --json
bigcherry results baseline --engine radiance --layout flashnext-4gpu --workload prefill-doc-44k --name reference --to reference.json
bigcherry results compare --baseline reference.json --engine radiance --layout flashnext-4gpu --workload prefill-doc-44k
bigcherry results history --engine radiance --layout flashnext-4gpu --workload prefill-doc-44k --metric prefill.prefill_tps --json
```

Each successfully completed workload publishes one immutable v1 run record.
The run folder contains a SHA-256 pointer. The record retains the exact
measured row as a provenance blob alongside original manifest bytes, engine
binary/build identity, native feature states, measured parameters, power,
environment, and host facts. Decode, prefill and TTFT are phase-qualified.
Read the record using `bigcherry results show SHA256 --json`.

The old JSONL index, its reader, seeder and workload-specific compare command
are not supported. Historical raw experimental artifacts remain as evidence,
not measured records. Without the original manifest and measured provenance
they cannot be reconstructed into trusted results. Do not use historical
run rows for gates: re-measure when a canonical record cannot be created
losslessly.

## Comparability

Use `bigcherry results baseline|compare|history`. A baseline requires multiple
launches and derives tolerances from measured spread, not a single request.
The platform retains phase directions, one-sided gates and provenance diffs
for layout, build, policy, power, kernel and host facts.

The document-slice prefill and fixed-prompt PWR phase rates are different
workloads. Stream-bench throughput is finite-batch makespan throughput, not
instantaneous generation speed. Cold TTFT is latency, not prefill throughput.

**Evidence rule:** cite workload ID/version plus immutable run-record SHA-256,
metric phase, provenance differences and gate verdict. Do not substitute a
discarded index row ID or an unrelated historical artifact.
