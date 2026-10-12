# Radiance platform design

Status: design for RAD01 / MEN03 / MEN04 / MEN06 / MEN07 / MEN08 integration.

## Inputs and observed state

This design is based on `main` at `a48c6968c1d57060b33053fd3e10bd4255357e9e` and Radiance work on
`feat/rad11-gfx1100-bootstrap` at `87106f51a3e473db908c07376e83f913b86e2c5e`.

Inferences / stale references:

- `docs/planning/active/patching-radiance-rdna3/RAD01.md` is not present on either inspected branch. The RAD01
  requirements in the task plus the feature-branch implementation are therefore the RAD01 design input.
- `tools/lab/plan-qualification/queue-env-ab.sh` is not present on `main`; `queue.sh`'s `AB` row and
  `bigcherry ab-benchmark --server-config` are the current A/B path.
- Runtime profile files have moved from `src/profile/*.ini` to
  `engines/llamacpp/overlay/profile/*.ini`. Separately, `config/recipes.toml` already has the platform-level
  `RuntimeProfile` object used by campaign/tuning launchers.
- MEN08 engine pillars and MEN03 process lifecycle have partially landed: `tools/bigcherry/core/engines.py`,
  both `engine.toml` files, and `ServerRunner(serve=ServeSpec,...)` already exist. Patch/source/build selection
  remains llama.cpp-rooted.

## Decisions

### 1. Engine patches are normal patch packages; do not add a git-diff package type

Move the three current diffs to:

```text
engines/radiance/patches/
  0001_drain_mover_before_arena_free/
  0002_moe_remote_handoff/
  0003_mtp_full_head_switch/
```

Each uses the existing `patch.toml` + `patch.py` + `SUMMARY.md` package contract. Existing-file changes use
anchored `FilePatch` edits. New files in 0002 use `FilePatch(create=True)` with an `\A` anchor and a unique
guard. `apply.py` already fails closed if upstream later creates that path, so the two additions
(`abi/rad_moe_remote.h`, `core/place/moe_remote.cpp`) do not justify an opaque `git apply` representation.

Do not preserve `engines/radiance/engine-patches/*.patch` or `engine-patched-build.sh` as compatibility paths.
Once the packages and standard build lane land, remove them. Package dependency edges are declared only where the
converted patches actually require one another; current all-or-nothing application is not evidence of dependency.

### 2. libr3 is a separately built engine component, not an overlay and not a vendor patch

Keep BigCherry-owned libr3 source under the Radiance pillar and give it an explicit component manifest:

```text
engines/radiance/components/libr3/
  component.toml
  CMakeLists.txt
  r3_compat.h
  r3_rewrite.py
  rewrites.txt
  native/*.rw
  ...
```

A component is source owned by BigCherry, built against a pinned engine source, and installed into the engine
runtime bundle without modifying `vendor/radiance`. This matches libr3: it consumes pinned `libr4d` sources,
rewrites a build-tree copy, emits `libr3.so` plus helper libraries, and is loaded through `RADIANCE_HOME`.

Do not copy libr3 into `engines/radiance/overlay/`: an overlay is materialized into the upstream source tree and
would falsely make the plugin part of Radiance source identity. Do not wrap changes to libr3 itself in patch
packages: component source is directly versioned BigCherry source. Patch packages remain transformations of the
pinned upstream engine only.

Enhancement needed: engine declarations/build plans gain named components and component content hashes. A build
that includes libr3 must bind the component digest and the Radiance pin in its build/runtime-bundle identity.

### 3. R3_* settings are runtime state, not patch lifecycle state

The component contains the implementation; an `R3_*` environment value selects a runtime form. Never model
`R3_ATTN_NATIVE=1` as “patch enabled” or change `patch.toml state` to perform an A/B.

Declare libr3 runtime settings in `component.toml` with type/default/diagnostic status so profiles and evidence can
validate them. Current disposition:

- `R3_ATTN_NATIVE`: supported/default-on in the component; profiles inherit it rather than restating it.
- `R3_GDN_NATIVE`: forced off; shelved/correctness-broken.
- `R3_MOE_NATIVE`, `R3_HC_NATIVE`: experimental until Flash-Next KL + ABBA qualification.
- `R3_AR_WIRE4`: experimental; requires its numeric/KL gate before use.
- `R3_AR_PIPE`: forced off; measured regression.
- `R3_AR_TIMING`: diagnostic-only, never production profile.
- `R3_NGRAM_DEV`: optional helper experiment; not part of the current two-XTX configuration of record.
- `R3_MOE_HELPER`: in-progress; no production profile until mechanics, correctness, KL and A/B evidence exist.

### 4. Runtime profiles are engine-owned recipe objects

Use the existing `RuntimeProfile` concept, but move engine-specific profiles into
`engines/<engine>/recipes.toml` with the engine's sources/builds/patch-sets/experiments. Extend RuntimeProfile with
`environment = { ... }`; keep ordered `server-args`. Do not add another INI loader to Radiance.

The two initial Radiance profiles are below. Owner rule (CLAUDE.local.md, 2026-10-05): a profile lists only the
settings that differ for that model or topology - component defaults (R3_ATTN_NATIVE=1, every other R3_* off) are
declared once in component.toml and never repeated in a profile.

```toml
[runtime-profile.flashnext-2xtx]
server-args = [
  "--tp", "2", "--tp-wire", "exact",
  "--placement", "expert_tiered",
  "--ngram-placement", "ram",
  "--host-pool-mib", "14336",
  "--max-num-seqs", "6",
  "--max-num-batched-tokens", "4096",
  "--checkpoint-interval", "4096",
  "--kv-cache-dtype", "fp8",
  "--num-speculative-tokens", "3",
]

[runtime-profile.qwen38-27b-2xtx]
server-args = ["--tp", "2", "--tp-wire", "wht6"]
```

The Flash-Next profile needs engine patch 0001 (the stager is armed by --max-num-seqs 6 and hangs at shutdown
without it). fp8 KV is what every measured run used; the owner's KV rule (q8_0 minimum, f16 preferred) is for
llama.cpp quantised caches - Radiance offers bf16 or fp8 only, and bf16 has not been fitted at this context yet.
The 27B two-XTX profile uses wht6: on the 27B it costs KLD +0.0008 / -0.4 pt top-1 (measured), unlike Flash-Next.

The 27B profile explicitly overrides only the two-XTX TP topology and measured `wht6` wire; its other settings
inherit engine/component defaults until a separately evidenced profile change is needed.

Flash-Next uses `--tp-wire exact`: the feature branch's KL run measured `wht6` at KLD 0.0746 / 87.18% top-1
against exact, so the earlier fast `wht6` setting is not a production-quality default.

## Engine-local recipe/config model

MEN08 owns the split. Final layout:

```text
engines/llamacpp/engine.toml
engines/llamacpp/recipes.toml
engines/llamacpp/patches/
engines/llamacpp/overlay/

engines/radiance/engine.toml
engines/radiance/recipes.toml
engines/radiance/patches/
engines/radiance/components/libr3/
engines/radiance/tests/

config/recipes.toml              # shared platforms/hosts/trees only
```

`engine.toml` becomes authority for upstream, pin, vendor path, serve protocol, build defaults and declared
components. `engines/<engine>/recipes.toml` owns patch sets, sources, builds, experiments and runtime profiles.
Every `Source` still carries `engine = "<name>"`; the loader validates that it matches the pillar containing the
file. There is no implicit/default engine.

Initial Radiance objects:

- sources: `radiance-native` (pin, no overlay/patches), `radiance` (validated Radiance patch sets), both
  `engine="radiance"`.
- patch sets: `serving-core` for validated engine fixes only; experimental 0002/0003 stay selectable through
  `[experiment.*]` until promoted.
- builds: `stock` and `release`, both building the declared `bin/radiance` target; component-bearing builds add
  `components=["libr3"]`. Diagnostic/selftest targets are explicit extra binaries, not hidden shell work.
- experiments: one focal engine patch or one named libr3 runtime switch per experiment. Do not make the current
  three-engine-patch stack an indivisible experiment.

## Required platform changes

### Engine identity and paths — MEN04/MEN08

`tools/bigcherry/core/engines.py`
- extend `EngineLayout` with pin/build/component declarations.
- keep `load()/load_all()` as the only engine declaration resolver.

`tools/bigcherry/core/paths.py`
- delete `LLAMACPP`, `PATCHES`, `SRC_OVERLAY`, and `llama_root()` as global authorities.
- replace callers with an explicit `EngineLayout` and `engine.vendor_root(primary_root)`,
  `engine.patches_root(repo_root)`, `engine.overlay_root(repo_root)`.
- migrate all callers in the same PR; no aliases or fallback default.

`tools/bigcherry/core/config.py::load`
- load shared `config/recipes.toml`, then every declared engine's `recipes.toml`.
- add `engine` to `Source`; namespace config identities by engine and reject cross-engine patch-set references.
- add `environment` and `components` where required.
- remove global `Config.pinned`; pin comes from `engine.toml`.

### Patch registry/composition — MEN08/MEN07

`tools/bigcherry/patch/registry.py`
- `PatchDescriptor` gains `engine`; canonical identity is `(engine, patch_id)`.
- `load_registry(engine, root, ...)`, `describe_all(engine,...)`, `load_implementation(..., root=...)` require
  explicit engine/root; remove defaults to `paths.PATCHES`.
- registry docs/errors stop naming `engines/llamacpp/patches`.
- dependency/conflict lookup is confined to the descriptor's engine and fails on cross-engine references.
- do not introduce a second registry or Radiance-specific patch dialect.

`tools/bigcherry/patch/apply.py`
- no engine-specific transformation logic is required. It already applies against a caller-supplied root and supports
  created files. Only evidence/result records need the engine identity from the caller.

`tools/bigcherry/patch/pinned_source.py`
- remove module-global `_VENDOR = paths.llama_root()`.
- make `copy_pinned`, `pinned_checkout`, `read_pinned`, `production_source`, `production_order` accept the
  engine/source selection explicitly; cache pristine checkouts by engine + resolved pin.
- remove hard-coded `source_name="bigcherry"`.

`tools/bigcherry/patch/{rebase,patchset,source,evidence,validation*}.py`
- thread engine identity through selection/evidence and use the selected engine registry/root. Patch identity stored
  in evidence is engine + id, not a bare globally interpreted id.

### Source/build lanes — MEN04

`tools/bigcherry/campaign/resolution.py::resolve_lane` and `ResolvedLane`
- derive engine from source; reject source/build objects from different pillars.
- selector/source identities include engine.

`tools/bigcherry/campaign/lane.py::CampaignLaneExecutionSpec`
- add/derive engine once at resolution and carry it through materialize/build/result provenance.
- materialization uses that engine's vendor/upstream/overlay/registry, never `context.patches_root` as an implicit
  llama.cpp root.

`tools/bigcherry/build/builds.py::BuildPlan`
- add `engine`, resolved engine pin, component ids/digests and target binary set to canonical identity.
- `build_directory()` becomes `work/builds/<engine>/<source_slice_id>/<build_plan_id>`.

Build worker/configure code chooses CMake source, target and options from the resolved engine/build object. The
default binary is `engine.serve.binary`; remove `bin/llama-server` assumptions.

`tools/lab/plan-qualification/queue.sh::build_line`
- call `bigcherry build --lane` without forcing `bin/llama-server`.
- consume emitted build metadata (`engine`, `build_plan_id`, binary path) rather than globbing
  `work/builds/*/<plan>/bin/llama-server`.
- keep shared GPU locks; build/runtime roots are partitioned by engine.

### Server and result normalization — MEN03/MEN06

`tools/bigcherry/tuning/server_runner.py` is already the right lifecycle abstraction. Do not create a parallel
Radiance runner.

- remove `LLAMACPP_SERVE`; every caller supplies the resolved `ServeSpec`.
- continue using declared health, shutdown, environment and draft-stat source.
- add `metrics_snapshot()` and log snapshot helpers so activation rules can use either without shell parsing.
- shared benchmarks use Radiance/llama.cpp OpenAI endpoints for cross-engine requests; engine-specific native
  endpoints remain explicit capabilities, not assumptions inside `ServerRunner.run_completion()`.
- result schema always records engine, resolved pin, source/build ids, runtime-profile digest, component digests,
  prompt/decode timings, drafted/accepted tokens and shutdown result.

The manual lifecycle in `libr3-serve-smoke.sh` is transitional and should be retired once its smoke/bench cases
run through ServerRunner.

## Radiance A/B and promotion flow

A Radiance campaign uses the existing queue + `ab-benchmark` path; an arm is
`(engine, source/build identity, runtime profile/settings)`.

For a focal patch or libr3 switch:

1. Mechanics: package/component lint, pinned applicability/rewrite check, compile, relevant selftest.
2. Build provenance: record Radiance pin, patch composition, libr3/component hash, CMake/compiler evidence.
3. Activation preflight: validation declares either a log regex or a metrics counter. Subject must fire/non-zero;
   control must be absent/zero. Add a low-cost Radiance marker/counter to a patch if its behavior is otherwise
   unobservable. This is patch-specific evidence, not a universal `BIGCHERRY_PATCH_HIT` requirement.
4. KL gate: run Radiance's native `--kld-record` reference from the exact control composition/profile, then
   `--kld-ref` on subject over the fixed corpus. Store mean/p99 KLD, top-1 agreement and perplexity ratio in the
   normal validation record. Thresholds live in the Experiment Contract. A failed KL gate blocks performance
   promotion.
5. Greedy identity/acceptance gate from normalized server results where the contract requires exact identity.
6. Decision-grade ABBA through `ab-benchmark`, complete arm separation, identical model/profile/hardware and clean
   process teardown between arms.
7. `patch-verify-evidence`/promotion consumes the engine-bearing evidence record; lifecycle promotion remains a
   separate metadata decision.

For runtime-only libr3 settings, steps 3-6 compare the same binary/component digest with only declared profile
settings changed. For upstream patch packages, control/subject differ by the focal patch composition as today.

The existing `fn-kld.sh` / `rad-kld.sh` behavior becomes a Radiance correctness adapter owned by the common
validation framework; those scripts cease to be the evidence authority.

## PR-sized implementation order

| PR | Scope | Key tests | Size | Closes / advances |
|---|---|---|---|---|
| 1 | Convert 0001-0003 to `engines/radiance/patches/<id>/`; mechanics only, no platform cutover | new `engines/radiance/tests/test_engine_patches.py`; each package applies to 89cee7c; created-file collision test; whole-tree compile check | M | RAD01 artifact packaging; MEN07 partial |
| 2 | Make registry/pinned-source engine-explicit; migrate every caller; no defaults/aliases | existing patch registry/rebase/offline suites + new two-engine registry isolation/cross-engine dependency rejection/pinned-cache tests | L | MEN08 patch pillar |
| 3 | Split recipes by engine and move pin authority to `engine.toml`; migrate all config callers | `tools/tests/core/test_config.py`; config identity tests; before/after llama.cpp production composition hash equality | L | MEN04 + MEN08 config/pin scope |
| 4 | Engine-aware source/build identity and queue BUILD path; remove llama-server target/root assumptions | campaign resolution/lane/build tests; BuildPlan engine collision test; queue shell syntax test; llama.cpp build-plan equivalence where semantics unchanged | L | MEN04 |
| 5 | Promote libr3 to declared component and integrate component build/runtime bundle | component schema tests; `r3_rewrite.py` exact-match tests; libr3 selftest build target; runtime-bundle component hash/reuse tests | M | RAD01 component integration; MEN01/RDNA3 build lane |
| 6 | RuntimeProfile environment + Radiance two-XTX profiles; ServerRunner metrics/log normalization | config RuntimeProfile tests; ServerRunner fake-process/fake-metrics tests; profile digest includes env; GDN/diagnostic production-profile rejection | M | MEN03 + RAD01 profiles |
| 7 | Engine-neutral activation + Radiance KL adapter in validation evidence | activation log/metric control-vs-subject tests; KL parser fixture tests; evidence schema engine/profile/component binding; failed-KL blocks promotion | M | MEN06 + RAD01 quality gate |
| 8 | Run one Radiance focal package end-to-end via queue/ABBA; retire `engine-patched-build.sh` and canonical manual A/B scripts | Brutus mechanics/build/activation/KL/greedy/ABBA record; `patch-verify-evidence` | M | MEN07 + MEN06; RAD01 complete |
| 9 | Move remaining Radiance lab evidence/runbooks into engine pillar and update references/release ownership | docs/link/path lint; CI path-filter tests; release dry-run if MEN09 not already complete | S-M | MEN08/MEN09 cleanup |

PRs 2-4 are platform migrations, not Radiance forks of existing logic. They should update all call sites in the same
change and delete the old llama.cpp-only authority instead of adding compatibility shims.

## Acceptance state after implementation

- A lane resolves one engine, pin, source, patch composition, build, components, platform and runtime profile.
- Radiance and llama.cpp use the same registry/composer/build/campaign/evidence machinery.
- Radiance engine patches are independently selectable/validatable; no all-or-nothing git-diff application path.
- libr3 has content-addressed component identity and is part of build/runtime provenance.
- Flash-Next two-XTX runs from the quality-qualified exact-wire configuration of record.
- Any Radiance performance promotion has mechanics + activation + KL/greedy correctness + isolated ABBA evidence,
  bound to the exact engine pin, patch/component digest, model and runtime profile.
