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
