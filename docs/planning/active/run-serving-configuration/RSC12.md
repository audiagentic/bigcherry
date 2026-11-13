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

1. **Format and loader (card-free).** `engines/radiance/configs/<model>.toml`, one file per model; `bc-serve` reads it
   in place of `--layouts` (`--configs FILE|DIR`, `BIGCHERRY_CONFIGS`). Strict: an unknown key, a key in the wrong
   section, a cycle or a missing base refuses, naming the file and the key.
2. **Variants.** `[config.NAME] base = "OTHER"`: the variant inherits every value of its base and states only what it
   changes; a chain is allowed (no cycles). Tables merge key by key; a scalar or a list replaces.
3. **Explain.** `bc-serve explain CONFIG` prints the effective values and, for each, where it came from (base config,
   variant, host file, `--set`, derived, detected).
4. **Convert the 13 layouts** into `configs/flashnext.toml` and `configs/qwen27b.toml`, with variants where a layout is
   a derivative (e.g. qwen27b: `tp3` base; `tp3-capacity`, `tp3-int8`, `tp3-int8-556` variants). Golden test: each
   converted config resolves to the same manifest digest as its layout did (fake facts, card-free).
5. **Stored manifest.** `bc-serve` keeps the manifest it writes for a config under
   `<run root>/manifests/<config>/<digest>.json` and reuses it while its key holds (config digest, host file digest,
   detected cards, engine identity, model identity, plan cache key); otherwise it writes a new one (measuring when the
   plan cache misses, as today). Free host memory is checked at every start against the manifest's ledger, not used as a
   key.
6. **Retire the old spellings.** `server-args`, `[layout.NAME.settings]` and the layout files go once every caller
   reads configs (RSC04 tracks the callers). `--set` stays for one-off experiments and is recorded in the manifest's
   provenance as `cli`; a setting worth keeping becomes a variant.

## Detailed Solution & Technical Design

File shape (values illustrative):

```toml
# engines/radiance/configs/qwen27b.toml
schema = 1

[config.tp3-int8]
description = "Qwen3.8-27B int8, XTX + XTX + R9700, tensor parallel"
model       = "radiance/bigcherry/qwen3.8-27b-w8a8/qwen3.8-27b-w8a8.rad"
cards       = ["gfx1100", "gfx1100", "gfx1201"]   # rank order; picked by architecture, never by index
tp_wire     = "exact"

[config.tp3-int8.service]
max_context  = 8192
max_sessions = 8

[config.tp3-int8.engine]
kv_cache_dtype         = "fp8"
num_speculative_tokens = 3

[config.tp3-int8.tuning]
tp_ownership      = "kv=1,1,2;ffn=6144,6144,5120;lk=6,6,4;lv=18,18,12"
pipelined_prefill = "on"

[config.tp3-int8-556]
base        = "tp3-int8"
description = "5,5,6 split: about 4x the KV room per card; passes the VTP36 quality gate"

[config.tp3-int8-556.tuning]
tp_ownership = "kv=1,1,2;ffn=5632,5632,6144;lk=5,5,6;lv=15,15,18"
```

Rules:

- Section and key names are the resolver's own keys (`service.*`, `engine.*`, `tuning.*`, `memory.*`, `helper.*`,
  `power.*`), the names `--set` and `bc-serve explain` use; no engine flag spelling appears in a config.
- Helpers are `[[config.NAME.helper]]` groups with the fields layouts have today (arch, count, budget_mib, placement,
  share). A budget left out is derived (RSC03).
- What belongs to the machine (ports, power profile, model root, run root) stays in the host's site file, never in a
  config, so a config is portable between hosts with the same card types.
- A config carries no derived values: host pool, host KV tier, checkpoint slots and helper budgets come from the plan
  (RSC03); the hand-picked handoff literals become config fields or derived values (RSC09).

The loader lives in the serving component (C++20, `engines/radiance/components/serving`) beside the resolver it
feeds. The Python tools that read layouts today (`bigcherry layout show`, campaign) read configs through
`bc-serve explain --json`, not a second parser (RSC05's one-inventory rule).

## Code Samples & Guidance



## Files

- `engines/radiance/components/serving/` (config loader, variants, explain provenance, stored manifests; tests)
- `engines/radiance/configs/flashnext.toml`, `engines/radiance/configs/qwen27b.toml`, `engines/radiance/configs/README.md`
- `engines/radiance/layouts/` (removed at step 6)
- `tools/bigcherry` layout readers (switch to `bc-serve explain --json`)
- `docs/reference/engines/radiance/SERVING_MANIFEST.md`, `RUNBOOK.md`

## Validation

- Card-free: loader refusals (unknown key, wrong section, cycle, missing base, duplicate name), variant merge, explain
  provenance, stored-manifest reuse and rewrite on each key input, golden manifest equality for all 13 converted
  layouts.
- Hardware (gpu lease): one dense config and one helper config start from configs through RSC02's `bc-serve start`
  and match the layout start's manifest digest.

## Effort & Risk

M. Risks: two input formats side by side during the move (bounded by the golden test and by deleting layouts at
step 6); a stored manifest reused when it should not be (every input that changes the plan or the processes is a key
input; the plan-key gap fixed in #959 is the case to test against).

## Standards

Architecture standards: config over code (no model or layout named in code), one resolver, strict validation.

## Acceptance Criteria

- Every model this repository serves has one config file; each layout of today is a config or a variant, with an
  unchanged manifest digest.
- A variant states only its differences; `bc-serve explain` shows where each effective value came from.
- No config or launcher contains an engine flag spelling; `server-args` and `[layout.NAME.settings]` are gone.
- A start of an unchanged config on an unchanged host reuses its stored manifest, and that file is the record of what
  ran.

## Notes

Consumed by RSC02 (`bc-serve start CONFIG`). Independent of the v2 contract cutover (RSC11 S13): configs are inputs
to the resolver; the manifest contract does not change. Owner direction 2026-11-14: one language for the start path
(C++, radiance's standard), clean readable configs, variants as named configs; GPT as backup reviewer and helper.

## Change Log

- 2026-11-13T22:56:54.680645+00:00 (created-by): Created by agent
