# Serving manifest: how one is made, how the engine starts from it

Operator guide for the Radiance serving manifest, edition 3 (`bigcherry/v3`). The contract itself is
[the spec](../../../design/serving-manifest-v2/spec.md) and its [schema](../../../design/serving-manifest-v2/serving-manifest.v3.schema.json);
every field and its type is in the generated [field table](../../../design/serving-manifest-v2/serving-fields.md). The configuration an
operator authors is one native Radiance YAML per serving config ([configs README](../../../../engines/radiance/configs/README.md), design:
[radiance-native-config-x-bigcherry.md](../../../design/radiance-native-config-x-bigcherry.md)); the manifest is what the owner writes from it.
This page is how to use them.

**One configuration method.** A manifest is the whole configuration of an engine or helper start: what to serve, how, and (for a
serving start) the memory plan it must admit. It embeds the exact bytes of the native YAML it was made from (`spec.native_config`:
the file, its SHA-256 and the resolution root), so a change to either half changes the plan key and the stored-manifest identity.
Direct engine flags (`--tp`, `--kernels`, `--no-prefix-cache`, ...) are retired in favour of it. There is no v2 reader, no TOML
catalogue of layouts and no `--set` on a managed start.

## Who does what

| Tool | Role |
|---|---|
| `bc-serve serve/start/stop/status` | Native production owner: reads the config's YAML, seals it, measures and validates the plan, owns engine/helper children and lifecycle; also `explain` and `quality`. |
| `radiance --manifest M --role engine` | The engine reads its whole configuration from `M`, which the owner hands it as a sealed memfd (`/proc/self/fd/N`). |
| `r3_moe_worker --manifest M --role helperN` | A helper reads its role from the same `M`. |

## How a manifest is made

The owner makes it; an operator does not write one by hand.

```bash
bc-serve start CONFIG --model-root DIR --run-dir DIR [--configs DIR] [--site SITE.toml] [--engine-binary PATH]
bc-serve start CONFIG --native-config NAME.yaml --model-root DIR --run-dir DIR     # a YAML file other than the bundled one
bc-serve explain CONFIG --native-config NAME.yaml --model-root DIR                 # the same admission, offline: no engine, helper or GPU
```

- `CONFIG` is the file name without `.yaml` in the verified bundle (or `--configs DIR`, `BIGCHERRY_CONFIGS`); `--native-config` names another file,
  which is sealed like any config. The site file keeps machine facts, power and calibration; it carries no engine options.
- The owner seals the YAML, asks the matched engine to inspect it (`radiance --bc-inspect-native-config`), reconciles the result with the
  `x-bigcherry` section and writes the manifest. `bc-serve --write-manifest` exists only as that internal step (it takes a private
  `--bc-native-config-fd`) and is not an operator interface. A manifest is immutable: an existing file is refused, and the result is read
  back through the same strict reader the engine uses.
- A serving manifest carries a **measured plan**. A named `serve` or `start` reuses the last measured manifest of its config from the work
  root's named-config store (`--work-root`; the key is the resolved settings, the ordered cards with their VRAM, the power profile, the config
  inputs and the model's path, size and mtime; the engine and helper builds are not part of it) and replays it without measuring. When the replay
  checks refuse the stored manifest because the engine or helper build changed, its measurement (plan, rank demand, resident host memory) is
  sealed again under the current build without running a measuring engine, and the old build is recorded as `plan_carried_from`. With no stored
  manifest, or none that can be carried, it measures first: the helpers come up on their cards, the engine runs to the end of its memory
  plan (`--plan-memory FILE --plan-memory-ranks FILE` on an unmeasured manifest) and exits, and the owner writes the measured manifest from
  that plan and stores it. An engine that does not come up on a stored plan drops the entry, so the next start measures. `start --manifest FILE
  --run-dir DIR` replays one named file. Run a measuring start under the `gpu` lease. An unmeasured manifest is accepted only by a measuring start.
- A serving start is production-only: it takes no lab experiment, live-policy override or calibration option. Only `bc-serve quality` takes a forced placement schedule (below).
- A lab arm that changes a setting authors its own YAML (`tools/lab/radiance/arm_yaml.py` writes a copy of a config with targeted changes);
  see [Campaign arms](#campaign-arms).

Inspect without starting: `bc-serve explain CONFIG [--json] ...` shows the effective intent of a config (see the [runbook](RUNBOOK.md)).

## Configs and engine-only manifests

A serving config names which cards run the engine and which hold experts, for one model, in `x-bigcherry.ranks` and `x-bigcherry.helpers`.
It is read at run time from the installed bundle or `--configs DIR`, so a model, split or topology is defined without a rebuild. Only the
card types (`engines/radiance/archs.toml`) are compiled in. The owner selects the cards by architecture from the host's detected GPUs; a
config never names device addresses. An engine may be a list of `{arch, count}` groups in rank order, a world of more than one card type.
A configuration with no host KV tier (`memory.host-kv-mib: 0`, the engine-only default) has a retention target of 0 tokens, and a dense
model is a valid engine-only model for the measured manifest.

| Config shape | Manifest |
|---|---|
| engine card + helper groups (the flashnext configs) | `processes.helpers` per helper, `handoff`, `helper_common`, a planned owner map |
| engine cards only (`qwen27b-2xtx`: two RDNA3 ranks, `tp-wire: wht6`) | `processes.helpers` and the owner map empty; **no** `handoff` and **no** `helper_common` |

For an engine-only config the engine's `tp` is the card count, `server.kernels` is the engine architecture's plugin plus `libref` (the mailbox
plugin `libr3moe` only when helpers exist), no handoff is needed, and the host-memory measurement is the engine's alone. A manifest with a
`handoff` or `helper_common` and no helpers, or helpers and no `handoff`, is refused by the reader. An engine-only manifest never reads the
model's expert tables, so a dense model (a container without routed experts, such as the 27B) gets one like any other; only a config with
helpers needs the experts, to plan which ones each helper owns.

## Starting with a manifest

```bash
radiance --manifest /proc/self/fd/N --role engine               # serving: the owner's sealed manifest, which must be measured
radiance --manifest M --plan-memory PLAN.json --plan-memory-ranks RANKS.json   # measuring start (unmeasured manifest file)
r3_moe_worker --manifest M --role helper0
```

A serving start takes the owner's **sealed memfd** (all four seals) named `/proc/self/fd/N` together with `--role engine`; a path on disk, a
missing role or an unsealed descriptor is refused. With `--manifest` the engine accepts only `--role engine`, `--plan-memory FILE`,
`--plan-memory-ranks FILE`, `-v`, `-vv` and the quality-run flags `--kld-record`, `--kld-corpus`, `--kld-ref` and `--kld-out` (a KL run scores
the model's own rows and never serves, so the serving plan check is not armed for it; a manifest with helpers is scored only by its matched
owner, `bc-serve quality`); any other flag is an error naming the flag. The engine's own options come from the admitted native YAML, parsed by
Radiance's parser; nothing of the manifest is projected to flags. A serving start checks the live memory plan against the plan the manifest
carries before any weights load; there is no switch to skip it.

The engine also applies the manifest's hand-off settings to its environment at start, before it initialises, so the manifest is the only place they are configured and they win over a value inherited from the calling environment: `R3_MOE_WORKER` (set when the manifest has helpers), `R3_MOE_WORKER_SHM` (`spec.handoff.transport.shm`), `R3_MOE_WORKER_MAX_ROWS` (`transport.max_rows`), `R3_MOE_WORKER_SKIP_OWNED` (`transport.skip_owned`), `R3_MOE_WORKER_HISTOGRAM` and `R3_MOE_WORKER_HISTOGRAM_PERIOD_S` (`telemetry.histogram_file`, `telemetry.period_s`). Kernel switches that were environment variables are manifest fields under `spec.processes.engine.kernels` (contract amendment 14): `attn_native` (`R3_ATTN_NATIVE`), `moe_native`, `moe_fp16`, `moe_ahalf` (`R3_MOE_*` in the engine process), `gfx12_moe_gemv` (`BC_GFX12_MOE_GEMV`), `all_reduce_wire4` (`R3_AR_WIRE4`), `dec11` (`R3_DEC11`), `gemv` (`R3_GEMV`) and `fuse_stage` (`R3_MOE_WORKER_FUSESTAGE`, amendment 16). `R3_MOE_WORKER_COMBINE` is set from `helper_common.combine.enabled` so the engine agrees with the helpers. Each is an `x-bigcherry.kernels` entry (`dec11: true`), and rad0009 sets the variable from the field before the engine initialises (generated from the registry's `env_setting`), so a value in the inherited environment loses to the manifest and an absent field leaves the kernel's own default. The lab-only patch and diagnostic-build switches are optional members of `spec.experiment.engine` (amendment 17): `routing_evidence` (`RAD_ROUTING_EVIDENCE`), `shadow` (`R3_MOE_WORKER_SHADOW`) and `mask_only` (`R3_MOE_WORKER_MASK_ONLY`), applied the same way and refused by the `production` profile.

## Where a setting lives

Every setting is authored once, in the config's YAML: Radiance's own options in its sections (`models.<id>.<key>`, `server.<key>`), and what
Radiance has no option for in the closed `x-bigcherry` section (`service`, `memory`, `handoff`, `helper`, `kernels`, `ranks`, `helpers`). The
[design](../../../design/radiance-native-config-x-bigcherry.md) maps every former resolver key to its place; the
[configs README](../../../../engines/radiance/configs/README.md) shows the file. A managed start refuses `--set`. A key derived from the
config or the host (`tp` from the rank count, `kernels` from the architectures, the runtime home, the host pool when the plan decides it) is
not authored.

The engine-side experiment fields of `spec.experiment` (lab profile only, written by the strict writer, never by a serving start):

| `spec.experiment.engine` field | Engine flag |
|---|---|
| `profile_ops` | `--profile-ops` |
| `kv_resident` | `--kv-resident` |
| `debug_selection` | `--debug-selection` |
| `deterministic` | `--deterministic` |

## Errors

A manifest that fails the strict reader is refused with a stable error code and the offending field
(`engines/radiance/components/serving` `errors.md`; the spec's error section). Native start reports typed serving refusals for canary, plan/admission and session ownership failures; the historical shell exit codes are not a supported interface.

## Campaign arms

A campaign arm that serves from a configuration names it under `serving`. The harness does not write a manifest for an arm: the owner does.

```json
"serving": {"config": "qwen27b-2xtx", "native_config": "tools/lab/radiance/27b-native-kld/attn-native.yaml"}
```

`config` is the serving config the arm's file is a copy of (it decides the model and the engine cards on this host) and `native_config` is the
arm's own YAML, written with `tools/lab/radiance/arm_yaml.py` (a repository-relative path is resolved from the checkout). A native arm takes no
`set`, no `server_args` and no `heat`; a changed value is a change in the file. It needs `execution_evidence: "observe"` and, today, a
`quality_only` job: the radiance adapter scores it with `bc-serve quality CONFIG --native-config YAML --model-root DIR --kld-ref DIR --kld-out FILE`,
which plans, seals and runs the finite KL score under the owner (helpers included). `bc-serve quality` only scores, so the job names a
stored reference (`quality.reference_dir` and `quality.reference_metadata`) rather than recording one; `quality.args` must be empty. See
`tools/lab/radiance/27b-native-kld.json`. Throughput arms that serve from a native YAML are not wired into the campaign harness yet.

### Forced placement in a quality score

`bc-serve quality ... --lab-experiment FILE` scores with a forced placement schedule (`serve` and `start` have no such option). FILE is the
JSON object the writer puts in `spec.experiment`, holding only `placement` with `mode: "forced_churn"` and a `forced` schedule; it is read and
checked whole before any plan (the manifest's own rules: a swap names both experts, a move names a destination helper, a step list of 1 to 1024,
`repeat` 1 to 1000000). The run measures afresh on the production plan, then writes a lab-profile manifest carrying the schedule (kept as
`lab-experiment.json` in the run dir); the named-config store is not read or written. Helper 0 issues one proposal per step once the engine
has completed its first lane request, so the swaps land during the scoring pass, not before it; `pace_requests` spreads them over completed
lane requests. When the score is over the owner asks helper 0 for its report and fails the score unless every transaction committed with none
aborted or in flight; the `forced_placement` event of the session record carries the counts and each resolved swap (layer, expert, destination
helper).

Every quality run (with or without the file) writes `owner-map.json` to the run dir before scoring: per helper its index, role, GPU, arch, tier,
formats, expert count and bytes, and `experts` as `[layer, expert]` pairs (the initial owner map; the layout is deterministic, so the control arm's
file serves for the arms that follow). A swap is legal when its two experts are on different helpers and each helper holds the other expert's
format: a tail helper (gfx1030) holds `w4nl64` only, so pick pairs whose experts are `w4nl64` on both sides; helper indexes in `move` and
`auto_swap` are the `index` of that file.

An explicit swap list (two transactions, repeated twice):

```json
{"placement": {"mode": "forced_churn", "forced": {"repeat": 2, "steps": [
  {"kind": "swap", "a_layer": 0, "a_expert": 3, "b_layer": 0, "b_expert": 9},
  {"kind": "swap", "a_layer": 0, "a_expert": 4, "b_layer": 0, "b_expert": 12}]}}}
```

`auto_swap` exchanges the lowest-numbered eligible (layer, expert) pair owned by helper `a_layer` and helper `a_expert` (here the two helper
indexes x and y, not a layer and expert), excluding the experts moved by the previous transaction:

```json
{"placement": {"mode": "forced_churn", "forced": {"repeat": 8, "pace_requests": 16, "steps": [
  {"kind": "auto_swap", "a_layer": 0, "a_expert": 2}]}}}
```

`move` (`a_layer`, `a_expert`, `destination`) sends one expert to another helper.

`R3_MOE_WORKER_*` arm settings, `R3_MOE_WORKER_LAYOUT` and `BC_SERVING_MANIFEST` are gone. Jobs that used them and could not be migrated
are archived in `experiments/radiance/archive-helper-env-jobs/` and `experiments/radiance/archive-engine-only-jobs/`.

## Migration status

| Step | State |
|---|---|
| Engine and helpers read a manifest; the engine admits the plan it carries | done (RSC06, RSC08) |
| One native YAML per config with a closed `x-bigcherry` section; the TOML catalogue, layout files and v2 readers removed; the manifest embeds the sealed YAML (edition 3) | done (RUM04) |
| Native owner: `serve`/`start`/`stop`/`status`/`explain`/`quality` from a sealed config, stored-manifest replay, session tests on the v3 fixture | done (RUM04) |
| Lab arms author their own YAML (`arm_yaml.py`); `27b-native-kld.json` runs through `bc-serve quality` | done in code; not yet run on hardware |
| Throughput campaign arms serving from a native YAML | pending |
| The engine reads its environment settings from the manifest | partly: kernel and hand-off switches are typed settings; the remaining `RAD_*`/`BC_GFX12_*` bridges stay until their consumers move |
