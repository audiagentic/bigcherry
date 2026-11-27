# Radiance serving configs

A serving config is **one native Radiance YAML file**, `engines/radiance/configs/NAME.yaml`; the file name is the config name
(`bc-serve start NAME`). There is no other catalogue: the RSC12 TOML files and the layout files are gone. Plan: RUM04, design:
[radiance-native-config-x-bigcherry.md](../../../docs/design/radiance-native-config-x-bigcherry.md).

```yaml
cards: [0, 1, 2]                    # Radiance: rank visibility order (a managed start requires [0..tp-1])
server:
  host: 127.0.0.1
  port: 36080
  kernels: libr3,libr4d,libref
models:
  qwen27b:                          # exactly one model
    model: radiance/StillDeadcode/qwen3.8-27b-iq8r-dflash/qwen3.8-27b-iq8r-dflash.rad
    tp: 3
    tp-wire: exact
    max-model-len: 262144           # context and sessions are authored here, once
    max-num-seqs: 8
    kv-cache-dtype: fp8
    tp-ownership: "kv=1,1,2;ffn=6656,6656,4096;lk=6,6,4;lv=18,18,12"
x-bigcherry:                        # what BigCherry owns and Radiance has no option for
  schema: 1
  power: {profile: efficient}  # named intent, not implicit application
  description: "..."
  variant-of: qwen27b-tp3      # optional: provenance of a variant, with differs
  differs: [models.qwen27b.max-model-len]
  ranks: [{arch: "gfx1100", count: 2}, {arch: "gfx1201", count: 1}]
  helpers: [{arch: "gfx1030", count: 1, placement: "tail"}]       # helper configs
  memory: {host-pool-mib: 0, host-kv-mib: 0}
  handoff: {heat: "@bundle/heat/flashnext-mixed.hot-first"}
  helper: {combine: true}
```

- **Two halves, one file.** Everything Radiance has an option for is in its own sections, parsed only by Radiance's parser (the
  engine's `--bc-inspect-native-config` returns what it read). `x-bigcherry` is skipped by Radiance and is a closed, typed schema
  (`schema: 1`: service, memory, handoff, helper, kernels, ranks, helpers) that bc-serve decodes from the same sealed bytes.
  A value is authored once: never both in the native section and in `x-bigcherry`.
- **Machine facts are not authored:** the runtime home (derived from the bundle or from `--engine-binary`), device addresses,
  host pool and KV tier (the memory plan unless `x-bigcherry.memory` overrides one). `server.port` is the one in the file.
- **No `--set` on a managed start.** A lab arm authors its own YAML (`tools/lab/radiance/arm_yaml.py` writes a copy with targeted
  text edits at named YAML paths) and passes it with `--native-config`.
- **Variants** are whole files that say what they change (`variant-of`, `differs`); a test over every pair fails when the files
  differ anywhere else.
- **Evidence** stays in comments above the key it justifies. Quote strings that YAML 1.1 reads as booleans (`"off"`, `"on"`).
- `bc-serve explain NAME --native-config NAME.yaml [--json]` runs the same admission as a start (inspect and reconcile with the
  matched engine) offline: no engine or helper child, no GPU. The released bundle ships every `NAME.yaml`
  (`tools/bigcherry/release/runtime_assets.py NATIVE_RELEASE_PROFILES`).

### Power profile

All shipped model configurations select `efficient` from the canonical
`engines/radiance/power-profiles.yaml`. A normal start is observe-only:
physical application still requires explicit `--allow-power-apply`,
verified lease, exact-device qualification and the shared transaction owner.
Do not use qualification mode as a production bypass. Stock remains available
by selecting `x-bigcherry.power.profile: stock` in a separate config or lab arm;
stock baseline and parity arms explicitly select it. No model weights, context
limits, kernel choices or memory controls change.


### Managed tuning enhancements (RSC19)

The native YAML remains authoritative for native engine options. BigCherry's
owned serving component projects values from the matched Radiance parser;
it does not parse native options again or implement a second scheduler.
Explicit native values override upstream defaults through the native parser.
New upstream native options should be inspected and qualified, not duplicated
under x-bigcherry.

The managed child environment does not inherit RADIANCE_ tuning, kernel
selection or diagnostic variables (nor the existing reserved RAD_, R3_, BC_,
BIGCHERRY_ and device-visibility settings). The owner supplies the selected
runtime home and device visibility. Use a non-reserved environment name for
an explicitly referenced API credential; unmanaged direct Radiance invocation
retains its own environment semantics. An approved lab arm authors native YAML
with the existing arm_yaml tool, rather than inserting ambient overrides.

The existing typed manifest fields for attention prefill precision, busy
prefill chunk, recurrent state precision and TP compression threshold now join
the native tuning projection. Reconciliation checks the presence and value of
all projected fields. The projection is not a new authoring schema. It records
authored intent; engine bring-up may still resolve defaults or refuse a
combination. The matched inspector v3 receipt also records parsed scheduler
interleave steps/share, mixed-step target, room gate, long-prefill threshold and
preemption wait, with explicit authorship flags. Omitted interleave, fixed zero,
`auto` (share 0.25), and `auto:1` are distinguishable. This is parse-time evidence,
not a claim of effective bring-up policy. The owner rejects malformed or
inconsistent receipts; effective-value reconciliation remains under RSC19.


### Tuning qualification and configuration evidence

VTP59/VTP60 tune native YAML arms using the existing arm_yaml writer, not
environment overrides. For a managed GPU measurement, capture
`capture_tuning_receipt` from `bigcherry.bench.radiance_lab_records` before
and after the workload. It binds native /diagnostics performance rows to the
admitted session, manifest digest, build and model/layout identity. Direct
manifest fields are reconciled automatically; the experiment must supply its
additional expected native diagnostic values (for example mixed-step target).
Missing, duplicate, contradictory or changed receipts are nonpass. The
existing native-lab ingestion accepts paired
`native-tuning-receipt.v1.json` and `native-tuning-receipt-end.v1.json` in
the run directory and retains them as evidence, without upgrading numerical
quality or attestation validity. Do not archive the entire diagnostics document.

This is a lab qualification seam, not a replacement for the serving owner's
admission or the planned matched-build runtime reconciliation. Dynamic defaults
and capability checks remain engine-owned. The matched owner now journals the parsed scheduler under session selection
with the sealed native YAML SHA-256, including on replay. The adapter derives
stable scheduler expectations from that receipt (pass an empty expectations
dict to use it); older owners need explicit arm expectations. Omitted interleave
is left to native bring-up. Qualify per model, topology and workload with
independent quality gates before production YAML changes.
