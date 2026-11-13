---
id: RSC12
order: 0
plan: run-serving-configuration
state: pending
created-at: '2026-11-13T22:56:54.680645+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Readable serving configs: one file per model, named variants, a cached manifest per config

## Description

Owner direction 2026-11-14: starting a model should be choosing one readable config, and a variant of a model's
config (another TP split, another context length) should be a named entry that lists only what it changes, not a copy of
a layout or a one-off `--set`.

Today (origin/main after #957/#959) the inputs are already run-time data (`engines/radiance/layouts/*.toml`, 13 layouts;
nothing compiled in), but they are not clean:

- one layout mixes raw engine flags (`server-args = ["--max-model-len", "8192", ...]`) with resolver keys
  (`[layout.NAME.settings]`), so the same setting has two spellings;
- a variant is a whole copy (qwen27b-tp3, -tp3-capacity and -tp3-int8 repeat each other) or a `--set` on the command
  line (the 5,5,6 split since #959);
- the bash launcher turns 15 options of its own into `--set` arguments (`tools/lab/radiance/lib/plan-check.sh`,
  `plan_check_describe`), a third spelling;
- the manifest is regenerated on every start, so no stored file says "this is what ran".

This item changes the **inputs** only. The manifest contract (v1 now, the single v2 cutover in RSC11) is unchanged: a
config resolves to the same manifest its layout does today.

## Steps

1. **Format, loader and adapter (card-free, inactive).** `engines/radiance/configs/<model>.toml`, one file per model,
   read by a strict `config_loader` in the serving component and translated by an adapter into the **existing** layout
   recipe (engine groups, helper groups, rank order, wire, model) and resolver settings. No second resolver, no change
   to the manifest contract; the active path is untouched until it is switched on. Strict: an unknown key, a key in the
   wrong section, a cycle, a missing base or a duplicate name refuses, naming the file and the key.
2. **Variants.** `[config.NAME] base = "OTHER"`: the variant inherits every value of its base and states only what it
   changes; a chain is allowed (no cycles). Tables merge key by key; a scalar or a list replaces (a variant that changes
   the helper groups restates the whole list).
3. **Explain.** `bc-serve explain CONFIG` prints the effective values and, for each, where it came from (base config,
   variant, host file, `--set`, derived, detected).
4. **Convert the 13 layouts** into `configs/flashnext.toml` and `configs/qwen27b.toml`, with variants where a layout is a
   derivative, keeping each layout's name as its config name until the v2 switch (S13) so `spec.intent.layout` and the
   plan keys do not move. Parity test: under the same fake inventory and host facts, each config and its layout give
   the same **semantic execution projection** (model, ordered engine and helper roles and cards, every effective setting,
   resources, owner map, handoff, memory-plan requirements), and both manifests pass the strict reader. Not a full
   manifest digest: every manifest carries a fresh `metadata.instance_id` and its provenance, both inside the digest.
5. **Stored manifests, three separate things (review P0):**
   - *static resolution key*: the canonical effective configuration (after variants, site, profile and `--set`) plus
     the identities of what it was resolved against (engine, helper and kernel binaries, runtime home, `bc-serve` itself,
     model artefact, heat file and lab experiment bytes, calibration records used, power profile, ordered cards with
     arch and VRAM). Equal key: the stored manifest's resolved content is reused.
   - *live admission on every start*: host MemAvailable, cgroup and memlock room, pin room, each rank's claimable VRAM,
     calibration validity, power and lease preflight. A stored manifest never skips these.
   - *execution record per start*: manifest digest, instance id, process identities, port, shared-memory namespace,
     applied power, canary and guard results, start and stop receipts.
   `start --manifest FILE` checks the stored bytes and digest without rewriting them, checks build, model and device
   compatibility and current admission, refuses with the specific reason, and records the run separately.
6. **Retire the old spellings** at RSC11 S16, not before: `server-args`, `[layout.NAME.settings]` and the layout files go
   once every caller reads configs (RSC04). `--set` stays for one-off experiments, recorded in provenance as `cli`; a
   setting worth keeping becomes a variant.

## Detailed Solution & Technical Design

File shape (the real qwen27b-tp3-int8 values are copied from the layout at conversion; the example shows the shape):

```toml
# engines/radiance/configs/qwen27b.toml
schema = 1

[config.qwen27b-tp3-int8]
description = "Qwen3.8-27B int8, XTX + XTX + R9700, tensor parallel"
model       = "<the layout's model path>"
ranks       = [{ arch = "gfx1100", count = 2 }, { arch = "gfx1201", count = 1 }]   # engine ranks in order, by architecture
tp_wire     = "exact"

[config.qwen27b-tp3-int8.service]
max_context  = 8192
max_sessions = 8

[config.qwen27b-tp3-int8.engine]
kv_cache_dtype         = "fp8"
num_speculative_tokens = 3

[config.qwen27b-tp3-int8.tuning]
tp_ownership      = "<the layout's split>"
pipelined_prefill = "on"
# ... every tuning value the layout sets today

[config.qwen27b-tp3-int8-556]
base        = "qwen27b-tp3-int8"
description = "5,5,6 split: about 4x the KV room per card; passes the VTP36 quality gate"

[config.qwen27b-tp3-int8-556.tuning]
tp_ownership = "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18"
```

Rules:

- Top-level config keys are the recipe: `description`, `model`, `ranks` (the engine's ordered `{arch, count}` groups), `tp_wire`,
  and `[[config.NAME.helpers]]` hardware groups (arch, count or "all", budget_mib, placement, share). Plural
  `helpers` for hardware groups, so it never collides with the `[config.NAME.helper]` settings table
  (`helper.combine`, ...).
- Settings tables (`service`, `engine`, `tuning`, `memory`, `helper`) hold the resolver's registered setting keys,
  the same names `--set` and `bc-serve explain` use, and feed its layout-settings layer; there is no flag spelling.
- Power intent (profile, role overrides) is its own `[config.NAME.power]` table mapped to `spec.intent.power`, not a
  resolver setting. The site file defines the host's power profiles and its default; a config may name one of those
  profiles (it wins over the site default), and `--set` wins over both. A config never defines a profile's settings.
- What belongs to the machine (ports, model root, run root, power profile definitions) stays in the site file.
- A config carries no derived values: host pool, host KV tier and checkpoint slots come from the plan (RSC03), handoff
  literals become fields or derived values (RSC09). A helper group's `budget_mib` may be given as an explicit override;
  left out, it is derived (RSC03).

The Python tools that read layouts today (`bigcherry layout show`, campaign) read configs through
`bc-serve explain --json`, not a second parser.

Known gap to fix first (review P0): the v2 manifest's embedded plan key (`v2/manifest.cpp` `plan_cache_key`) omits the
engine tuning settings and keys only the first engine card, so a changed `tuning.tp_ownership` on a helper-free TP
layout leaves it unchanged; the native key was fixed in #959. One shared, versioned fingerprint for both keys, covering
every rank's card and every plan-affecting setting.

## Code Samples & Guidance



## Files

- `engines/radiance/components/serving/` (config loader, variants, explain provenance, stored manifests; tests)
- `engines/radiance/configs/flashnext.toml`, `engines/radiance/configs/qwen27b.toml`, `engines/radiance/configs/README.md`
- `engines/radiance/layouts/` (removed at step 6)
- `tools/bigcherry` layout readers (switch to `bc-serve explain --json`)
- `docs/reference/engines/radiance/SERVING_MANIFEST.md`, `RUNBOOK.md`

## Validation

- Card-free: loader refusals (unknown key, wrong section, cycle, missing base, duplicate name), variant merge, explain
  provenance, stored-manifest reuse and rewrite on each static-key input, semantic execution projection equality for
  all 13 converted layouts (step 4), byte-for-byte replay of a stored manifest (`start --manifest`).
- Hardware (gpu lease): one dense config and one helper config start from configs through RSC02's `bc-serve serve`
  and give the same semantic execution projection as the layout start.

## Effort & Risk

M. Risks: two input formats side by side during the move (bounded by the projection parity test and by retiring
layouts at S16); a stored manifest reused when it should not be (the three-way split above: anything that is not a
static identity is checked live at every start); the model identity (`plan_cache.cpp` model_identity) hashes size,
header and tables, not every tensor byte, so the static key uses a full artefact digest where exact weights matter;
concurrent edits of `serving.cpp`, `manifest_writer.cpp` and the generated registry (RSC11's single integrator owns
those; the loader and adapter land in their own files).

## Standards

Architecture standards: config over code (no model or layout named in code), one resolver, strict validation.

## Acceptance Criteria

- Every model this repository serves has one config file; each layout of today is a config or a variant with the same
  semantic execution projection (exact digest equality only for the replay of the same stored manifest bytes).
- A variant states only its differences; `bc-serve explain` shows where each effective value came from.
- No config or launcher contains an engine flag spelling; `server-args` and `[layout.NAME.settings]` are gone.
- A start of an unchanged config on an unchanged host reuses its stored manifest, and that file is the record of what
  ran.

## Notes

Consumed by RSC02 (`bc-serve serve|start CONFIG`); RSC02's engine-only lifecycle can start on today's layouts, so
neither blocks the other. Independent of the manifest contract: configs are inputs to the resolver; nothing here
activates v2 ahead of RSC11 S13.

First PR (card-free): `config_loader` + variants + adapter + one dense example and its variant
(`qwen27b-tp3-int8`, `-556`) + the projection parity test against the layout. No start/stop, no stored manifests, no
caller migration, no layout deletion.

Owner direction 2026-11-14: one language for the start path (C++, radiance's standard), clean readable configs,
variants as named configs; GPT as backup reviewer and helper. GPT review 2026-11-14 (req_rsc02-config-launcher_957d54878bcc):
three P0 (owner process lifetime, the two plan keys, digest-equality gate) folded in above and in RSC02.

## Change Log

- 2026-11-13T22:56:54.680645+00:00 (created-by): Created by agent
