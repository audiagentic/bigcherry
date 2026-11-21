# serving

## Summary

The `bc_serving` C++ library and `bc-serve` tool provide a strict typed
TOML loader, hardware detection, configuration validation and a canonical
serving manifest with provenance.

## Outcome

The strict TOML loader, `bc-serve` inspection and manifest generation,
the shared v1 reader, and the engine-side memory-plan measurement and
admission check are implemented. The serving component now includes an
owner-map verifier that checks the planner's owned expert sets and exact
device byte totals. Card-free tests cover the verifier. The helpers read their
whole configuration from the manifest (`r3_moe_worker --manifest FILE --role
helperN`, through `v1/helper_settings.h`; RSC06) and the launcher writes the
manifests they start from. The engine still starts from flags until the S4
cutover, so this is not yet a fully qualified v1 launch.


## Native YAML candidate

Build the patched `radiance` source with the `stock` recipe: the engine, `bc-serve` and
the MoE worker all use wire edition 3. Each build generates
the codec from the same registry and verifies the tracked fingerprint.

For a fresh v3 start, select the BigCherry hardware profile and author the
Radiance settings in YAML. Relative YAML paths resolve under the model root.

```sh
bc-serve start CONFIG --native-config fleet.yaml --model-root MODEL_DIR --run-dir RUN_DIR
```

For an exact replay, use `bc-serve start --manifest MANIFEST --run-dir RUN_DIR`.
It reads the YAML bytes and resolution root from the admitted manifest.

## RSC02 step 3: manifest-bound power and native runtime guards

The foreground `bc-serve serve` and detached `bc-serve start` owner authenticate
the pre-measurement `v2::Manifest` with the strict reader and bind power to its
ordered engine/helper PCI addresses. On a replay of a stored measured manifest (a named start that reuses one, or `start --manifest`), that manifest is
the initial power admission; on a miss, the unmeasured manifest is admitted
**before** any helper or engine measuring process. Both are journaled and
the final measured manifest is checked against the admitted role/address order,
profile digest and tuning. An owner that fails to record admission durably
cannot launch a measuring child. The owner reaps all registered child
process groups before calling `SessionPower::finish()`; every returned power
or receipt error surfaces as a failed session, never a clean stop.

The default `--power-profile stock` is observational. Hardware-changing
power profiles that change hardware require **both** a qualified profile and the opt-in
`--allow-power-apply`, plus an unexpired, ancestor-and-instance-bound
GPU lease. With no opt-in the owner rejects apply before a hardware write.
No unit test calls the real Linux write interface: power tests use injected
`power::Hardware`, `power::Records` and `power::Sensor`.
PWR06 captures a no-process floor after measurement children exit, before
the serving helpers and engine start; it captures loaded-idle evidence after
health and before requests, evaluates the typed `power::Verdict`, and persists
the advisory result in the session power receipt. Missing sensors cannot
be misreported as healthy GFXOFF.

A successful canary is followed by a **manifest-driven** MTP replay guard
(depth greater than one, absent `profile_ops`): an engine log must include a
recorded nonzero draft pass, with exactly one retry canary before refusal.
The heat-stream guard reads the unchanged engine histogram producer layout:
native-endian cumulative `uint32_t[128][model.experts]`, including zero-padded
inactive model layers. Its model dimensions are authenticated from the strict
manifest and must fit the common `moe_shm::kMaxLayers` and expert-count domain.
Each publication is an atomic replacement; the guard accepts new inodes when
the full record and all per-cell counters remain consistent, and fails closed
on resets or counter wrap rather than fabricating positive deltas.

A serving owner mints a 128-bit nonce immediately before spawning each engine
and passes the launch-only `--bc-heat-launch-nonce` flag to only that engine.
The strict manifest reader validates and strips the flag, then binds its value
in the immutable `BcEngineSettings` snapshot, never the environment. The
engine-side histogram publisher retains the *exact* raw `.rankN` layout for
the worker reader and optionally writes a versioned `.rankN.bcproof` companion
containing that nonce and the published file's generation. The owner requires
a matching proof and a post-launch, fresh, nonzero sample for every successful
poll; it never treats an unproved legacy rank file as authenticated session
evidence. Standalone engines without the nonce remain backward-compatible
with existing raw-histogram workers. Since the two renames are separate,
the owner retries a bounded number of times over a transient mismatch.
This token guards against stray and previous-launch producers, not a
same-UID adversary capable of inspecting the child environment or forging the
publication. The sidecar is not a signature or an authority over arbitrary
writers with access to the owner's credentials.

For a manifest requesting live policy, the C++ runtime guard verifies the
startup decision, samples the actual rank histogram without following
symlinks, rejects empty/stale/future files, and checks live worker transport
identity and placement journals through the typed runtime receipt reader.
A live degradation or stale transport fails the owner; polling is bounded.

Foreground lab calibration is opt-in:
`bc-serve serve --calibration-run --live-policy FILE
--calibration-prompts-a FILE --calibration-prompts-b FILE
--calibration-store PATH`. It needs an authenticated measured lab manifest
with helpers and no experiment, a finite nonempty A/B/B/A workload, measured
drift evidence before the prefeed deadline, and a single complete calibration
marker. A capped observation-space envelope prevents an unbounded run; the
existing C++ ingest validator rejects malformed or incomplete records.
No successful live-policy state is reported merely because calibration was
requested; a final runtime telemetry proof is required. Detached calibration,
automatic activation of a partial calibration and arbitrary lab shell scripts
are not offered.

Card-free tests: `bc_manifest_power_test`, `bc_session_guards_test` and
`bc_session_test` (as well as existing `bc_power_transaction_test`).
Every executable is a CMake target and declared in `component.toml`.

## MRC03 native feature states

At startup, after the exact measured or sealed strict manifest has been
selected and before serving children launch, the `bc-serve` owner serializes
`run_dir/feature-states.v1.json` (platform v1 JSON array). Its seven
manifest-derived entries cover the live placement policy and its startup
decision/provenance skip reason, MTP depth, prefix cache, host KV tier, helper
combine, vision tower placement and maximum image patches. Each entry names
the engine, observation time, state and reason; this is a configuration/start
snapshot, **not** evidence that the engine's runtime activation was verified.

The same entries are saved in `session.json` and returned by
`bc-serve status --run-dir DIR --json`. For helper-backed production starts,
inactive/degraded live placement emits a single-line `WARNING production
feature live_policy` with reason code and detail to the owner log. An
engine-only or otherwise not-applicable policy does not warn. The snapshot is
archived on a subsequent owner start so an old run cannot masquerade as a
new startup decision. Other optional features are reported without
implicitly being declared production-required. The native mapping and
file/status/warning paths have the card-free `bc_feature_state_test`
and `bc_session_test` CTest targets.

## Callback boundary and source scanner

Injected callbacks in the current serving component use private typed
`callback::Guarded` / `GuardedSymbol` storage or private virtual methods
dispatched by guarded adapters. The callback source scanner (`tools/tests/core/serving_callback_guard.py`, a platform test: the component is C++ only) is a **best-effort pattern tripwire, not proof** that a future
callable type cannot bypass this boundary.

REVIEW4 found two compiling **known gaps** (both missed by the scanner,
neither a passing safety test):

```cpp
int callback_impl(int) { throw 17; }
using Hook = decltype(&callback_impl);
struct Owner { Hook callback; int invoke(int x) { return callback(x); } };
```

```cpp
class Injected { public: virtual int callback(int) = 0; };
class Throwing : public Injected {
  public: int callback(int) override { throw 17; }
};
int use(Injected& cb) { return cb.callback(3); }
```

The private wrappers protect current injection sites; the scanner alone
cannot inventory arbitrary aliases, layouts, virtual interfaces or
indirect calls. [RSC13](../../../../docs/planning/active/run-serving-configuration/RSC13.md)
plans a clang AST inventory and CI gate with negative counterexamples.

## Detail

CPU-only static library `bc_serving` and `bc-serve`, packaged at
`radiance_home/bin/bc-serve`. One C++ schema and strict TOML loader;
no Python schema or shell launcher parser; session and helper/engine process ownership
are native C++.

Build with `-DCMAKE_PREFIX_PATH=<installed radiance>` and
`-DRADIANCE_SRC=<radiance source>`; build `bc-serve bc_serving_test`
then run `ctest --output-on-failure`. Tests are CPU-only, including a
synthetic KFD/PCI/proc/sysfs tree and launcher-value parity. The resolver tests are engine-free: they select a
config through a C++ preset (`test_config_fixture.h`), generated from the shipped `engines/radiance/configs/NAME.yaml` by
`gen_fixture_presets.py` (a test fails when they drift); the production resolver reads a config only from its YAML.

Commands: `--detect [--json]`, `--schema`, `--validate <site> [profile]`,
`--explain [--json]`, `--write-manifest OUT` (the serving manifest v1; with `--plan-cache` and `--engine-binary`,
or with `--unmeasured` and neither: the measuring start's manifest, `plan.state` unmeasured, which only a measuring start
(`radiance --plan-memory`, `r3_moe_worker --measuring`) accepts; `--live-policy FILE` adds the live placement policy and
`--experiment FILE` a lab experiment, each a JSON object the reader checks member by member; `--measured-plan FILE
--helpers-resident-mib N --engine-resident-mib M` takes the plan a measuring start produced and the host memory the helpers and
the engine held in it, checks the plan against the configuration, stores both in the plan cache under this configuration's key
and writes the measured manifest from it; the host ledger counts the host pool and those two figures against MemAvailable as
read when the manifest is written, so the checkpoint slots follow what is really free at start, not at measurement; a cache
entry without the figures is a miss). The engine admits the plan the manifest carries; no other
command or flag hands it one (RSC08 S3 removed `--plan-key`, `--plan-check-args` and `--plan-store`).
Use `--site FILE`, `--profile FILE`, and repeated
`--set section.key=value` with the latter three.
`--sys-root DIR` overrides the filesystem root for fake-tree tests.
Site-only `[hard_caps.memory]` defines maximum MiB; caps may only
tighten the discovered/default baseline. Neither profiles nor CLI may
modify hard caps. Unknown keys fail closed with their dotted path,
source file and closest valid key.

Schema:
- `service`: objective, max_sessions, max_context, retention, quality
- `layout`: name, engine_card (PCI BDF), helper_cards (PCI BDFs)
- `memory`: host_pool_mib, host_kv_mib, tail_budget_mib,
  helper_budgets_mib (in helper order)
- `handoff`: shm, max_rows, small_rows, gemv_large_rows, moe_down_tile,
  skip_owned, heat, histogram, histogram_period_s, placement; these are
  the launcher's COMMON mailbox settings, not WorkerConfig fields
- `helper`: **exactly** the ten WorkerConfig fields: count, index,
  engine_ranks, cpu, power_save, queue_warm, combine, combine_min_rows,
  combine_threads, cpu_set_auto
- `engine`: model, radiance_home, port, kernels, placement,
  ngram_placement, host_pool_mib, prefix_cache_mib, tp, max_num_seqs,
  max_num_batched_tokens, checkpoint_interval, kv_cache_dtype,
  num_speculative_tokens, max_model_len, rope_scaling, prefix_resume

WorkerConfig's index is expanded per helper in `process_helpers`.
The `engine.model`, `engine.radiance_home`, `handoff.heat`, and
`handoff.histogram` fields have no portable defaults and are required.
Use `handoff.histogram = "off"` to disable recording. Values for heat
and histogram should be supplied as paths from the site/profile.
For example, for a read-only explanation:

```
bc-serve --explain --set engine.model=/models/model.rad \
  --set engine.radiance_home=/runtime \
  --set handoff.heat=/data/heat.hot-first \
  --set handoff.histogram=/data/heat.live
```

The JSON Schema `x-origin` annotations document every default source.
Interim memory baselines and named layout GPU budget baselines come from
the 2026-11-02 launcher; RSC03
will derive them from detected resources. A first-slice snapshot of the
named layouts is used to select cards from live KFD discovery rather
than reading any per-host configured GPU index list. Unsupported layouts
and unavailable GPUs fail closed; layout/catalog generalization is later
RSC05 work.

The one document processes load is the serving manifest (edition 3, written by the owner from the config's sealed native YAML; spec in
`docs/design/serving-manifest-v2/`), read by the shared strict reader. `explain`'s record is not a file any process reads. The engine and
helpers start from that manifest (rad0009, shared `bc_serving_v2` library).
The owner now launches and verifies measuring/serving process groups;
legacy shell and Python paths remain reference oracles until the later
hardware A/B and retirement gate.

Vendor: toml++ v3.4.0 root single header and MIT license (`third_party/`; the repository ignores every `vendor/` directory);
reuse radiance's existing nlohmann/json v3.12.0.

## RSC02 step 6: native named-config inspection (RSC12)

`bc-serve explain CONFIG --configs engines/radiance/configs --site SITE --model-root ROOT [--set key=value] [--json]`
inspects the effective RSC12 named configuration using the **same native
detector, config loader and resolver** as `serve/start CONFIG`. It starts no
children, does not measure a plan and requires no GPU lease. In text mode it
prints engine/helper counts and each provenance record; JSON includes
`config`, `values`, `provenance`, `hash` and the resolved template
`manifest`. An unknown config, unknown option, extra positional argument or
`--set layout.name=...` override is refused rather than silently selecting
another layout.

**Design source:** RSC12's one named configuration plus provenance and RSC02's
single native resolver. **Difference from the retired interfaces:** this is
not a Python `bigcherry serve explain` wrapper and does not derive defaults
from the bash launcher's option parser. The existing `bc-serve --explain`
remains a lower-level, site/profile-oriented resolver inspection; no
launcher-compatibility flags are introduced. `bc_serving_test` execs the
actual `bc-serve` binary with a synthetic KFD/PCI/proc filesystem and a real
config file, checks base/variant provenance, JSON/text behaviour and refusal
cases without requiring any GPU, permission failure or fake planner.

## RSC02 step 5: named configs and immutable stored replay

A foreground owner can start an RSC12 config with `bc-serve serve CONFIG --run-dir DIR
--configs engines/radiance/configs` or `BIGCHERRY_CONFIGS` set to the config file/directory.
`bc-serve start CONFIG` detaches the **same** owner. A named start reuses the
last measured manifest of that config when the work root (`--work-root`) holds
one, whichever run directory wrote it and whichever build runs it; otherwise it
measures, writes a strict manifest and stores it as the config's entry in
`<work-root>/cache/named-configs/`. The reuse key names only what decides the
plan: the resolved effective settings (without the installed runtime path, the
per-start transport name or the derived host pool), the ordered cards with
their architecture and total VRAM, the power profile, the config/site/heat/
calibration inputs and the model's canonical path, size and mtime (the model is
never read). The engine, helper and bc-serve builds are not part of the key.
The engine's own admission is the safety net: a stored plan the replay checks
refuse (for example a different engine build) is measured once afresh and
replaces the entry; an engine that then fails to come up on a stored plan
removes the entry so the next start measures. Without `--work-root` every start
measures. The old `--set layout.name=...` serving path remains on its original
planner.

`bc-serve start --manifest FILE --run-dir DIR --configs FILE|DIR` replays a
previously recorded strict manifest **without writing or normalising it**.
A writer-produced manifest carries a separate atomic `FILE.replay-index.json`
attestation (its original byte fingerprint, model stat identity
and RSC12 static key); the manifest bytes remain immutable. The RSC12 config
sources are needed for a fresh effective-settings and calibration check.
A missing attestation or config source, changed input, changed build/model/
selected device, invalid digest, stale calibration or insufficient current
RAM/cgroup/memlock/pin/per-card VRAM stops before any GPU child. The owner
still performs the authenticated power/lease admission, durable process
record, start health/canary, runtime guards, verified shutdown and power
restore. The session record is per start; it is **not** the manifest or the
cache index. The original bytes are checked again immediately before
launching a serving child.

An explicit `--power-profile` conflicting with the recorded profile is
refused. A stored lab experiment is not enabled by default: replay lacks
the RSC12 lab-experiment static input-key oracle until that part lands,
and refuses rather than inventing a config format. RSC12's dedicated
`[config.NAME.power]` and site-defined profile tables also remain a
loader dependency: this step binds the existing explicit owner
`--power-profile` mechanism but does not introduce an unofficial power
table schema. RSC02 step 6 adds a separate, explicit **foreground lab observation** mode:
`bc-serve serve CONFIG --configs DIR --run-dir DIR --lab-experiment FILE`.
FILE is parsed as a lab experiment object (`spec.experiment`); only a placement object with
`mode: observe` (and no forced schedule) is accepted in the actual
`plan_engine_only` path, before the writer or any GPU child. The C++ owner
reuses the production named resolver, measuring owner, strict writer/reader,
durable power admission, runtime guards and verified stop. The experiment
does not enter the stored-manifest index or replay. RSC12 has not yet bound
shape-changing experiments into both the measured plan and static manifest
key, so shape, forced churn, profiling and mixes with calibration/live policy
fail closed. This is deliberately **not** the Bash `--experiment` forwarding
surface: no shell environment transport, old alias or Python launcher is
included. A fake-device CTest exercises the real resolver-to-writer bytes
and refuses unsupported variants; hardware S15 is still pending.

The bash launcher/Python supervisor are unchanged pending
RSC02 step 6.

CPU-only acceptance: `bc_stored_manifest_test` (strict bytes and typed
identity refusals), `bc_stored_manifest_cache_test` (static reuse and
invalidation), `bc_session_stored_test` (real owner call graph with fake
GPU processes/OS facts). The dependency probes used by tests are explicit
C++ callbacks; production always uses `detect`, the single RSC12 resolver,
the strict manifest reader and P1 live-room measurement.

## RSC02 step 6: typed lab inputs, not shell flag aliases

The RSC12 native registry already owns the launcher-mode values:
`engine.port`, `handoff.heat`, `handoff.histogram`,
`handoff.histogram_period_s`, `engine.kv_cache_dtype`,
`engine.ngram_placement`, `engine.num_speculative_tokens`,
`service.max_context`, and `memory.host_pool_mib`/`host_kv_mib`/
`helper_budgets_mib`. Explicit helper budgets belong in
`[[config.NAME.helpers]] budget_mib`. Named variants, site files, and
recorded `--set key=value` values are resolved with source provenance;
`--ngram-device` has no typed contract field and remains refused.

**Design source:** RSC04's mode table, RSC12 named-config precedence,
and the native serving manifest process contract. **Difference from the bash
launcher:** no second syntax translates `--port`, `--heat`, `--budget`,
or other legacy flags into environment variables; the one resolver writes
the typed manifest (edition 3, sealed with the native YAML). `bc_typed_settings_test` executes the compiled `bc-serve
explain` and `--write-manifest --unmeasured` against a synthetic KFD/
PCI/proc filesystem, real named configs, and producer-shaped route heat,
checking provenance, manifest fields and typed refusal cases. The test's
unmeasured manifest is not serving authorization. GPU parity remains S15.
