# serving

## Summary

The `bc_serving` C++ library and `bc-serve` tool provide a strict typed
TOML loader, hardware detection, configuration validation and a canonical
serving manifest with provenance.

## Outcome

The strict TOML loader, `bc-serve` inspection and manifest generation,
the shared strict manifest reader, and the engine-side memory-plan measurement and
admission check are implemented. The serving component now includes an
owner-map verifier that checks the planner's owned expert sets and exact
device byte totals. Card-free tests cover the verifier. The engine and the
helpers read their whole configuration from the serving manifest the owner
writes from the config's native YAML (`r3_moe_worker --settings-fd FD --role
helperN`: typed `v2/helper_settings.h` on a sealed descriptor; RSC06). Planning is native-only: a fresh start
needs an admitted native config YAML, and there is no other planning path.


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

## RSC02 managed native entry (rad0070): how the owner starts every engine

The owner starts the stock `radiance --config FILE` and adds `--bc-admission-fd FD`: one sealed, strict JSON document
(`v2/engine_admission.h`) with only what the owner derived (ordered rank BDFs, host pool/KV/reserve/checkpoint sizes, the
rad0011 expected plan and KV floor, the typed engine/remote settings, the not-yet-typed engine switches). It is not a copy
of any native setting. After the YAML is parsed, `bc_serving_managed_bind` applies each derived value to an absent or auto
option, accepts an agreeing authored one and refuses a disagreement naming both; one model, loopback, no per-card headroom
map. Fleet's usable cards are the admitted ranks (Radiance's own filter keeps one architecture and would refuse mixed TP; the owner has restricted visibility to exactly those cards), and its chosen cards must be that order before the Engine exists. A managed process refuses SIGHUP and
`/admin/reload` and model admin (403) and ends, failing, when its engine stops. Without the flag nothing changes. Card-free
tests: `bc_native_managed_test` (engine build), the admission wire in `bc_serving_v3_wire_test`.

The owner starts the measuring engine, the serving engine, a replayed engine and an offline KL run this way
(`engine_inputs` in session.cpp: the YAML of the admitted manifest sealed as `--config /proc/self/fd/N`, the admission
sealed beside it, the measuring flags and the heat/phase flags after them). The manifest stays the owner's own record
(power, guards, replay) until RSC11; no engine reads it. A KL run's files travel in the admission (`quality`), not as
flags. Clients that were not told a model ask `GET /v1/models` (`bigcherry.bench.served_model`); the implicit
"default" alias is gone.

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

The power profile is the config's `x-bigcherry.power.profile` (a profile of `engines/radiance/power-profiles.yaml`, compiled
into bc-serve and read through the engine's YAML parser; `stock` when absent) and is observational. Hardware-changing
power profiles that change hardware require **both** a qualification and the opt-in
`--allow-power-apply`, plus an unexpired, ancestor-and-instance-bound
GPU lease. With no opt-in the owner rejects apply before a hardware write.
The qualification is the controller's own durable receipt (`<work-root>/power/receipts`): a setting is accepted for a
BDF when an earlier apply of exactly that setting on the same kernel, amdgpu version and board (the receipt's plain
`identity` string, compared by equality) read back exactly and restored exactly; the receipt's workload is evidence only.
The first such apply is the lab run `--allow-power-apply --qualify-power` (same lease; it waives only the receipt match,
never the probed range or the gfx1100 pairing rule). The capability probe is the transaction's own read of the card. The S10 output (the typed manifest) goes straight to the
power transaction; power is admitted once, before any measuring child, and the final serving manifest must carry the same
effective settings. The calibration key's `power_profile` is the profile name, or `name@digest` of the effective per-BDF
settings when the plan applies.
This host's per-BDF settings come from `<work-root>/power-site.yaml` (`schema: 1`, `devices: {BDF: {settings, evidence}}`,
read through the engine's YAML parser) and replace the profile's settings for that card only.
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

Card-free tests: `bc_manifest_power_test`, `bc_session_guards_test` and
`bc_session_test_part1` to `_part4` (the session tests, in four groups; as well as existing `bc_power_transaction_test`).
Every executable is a CMake target and declared in `component.toml`.

## MRC03 native feature states

At startup, after the exact measured or sealed strict manifest has been
selected and before serving children launch, the `bc-serve` owner serializes
`run_dir/feature-states.v1.json` (platform JSON array). Its seven
manifest-derived entries cover the live placement policy and its startup
decision/provenance skip reason, MTP depth, prefix cache, host KV tier, helper
combine, vision tower placement and maximum image patches. Each entry names
the engine, observation time, state and reason; this is a configuration/start
snapshot, **not** evidence that the engine's runtime activation was verified.

A start of a named config says whether it reused the stored plan, in one stderr
line: `bc-serve: stored plan reused (...)`, or `bc-serve: no stored plan reused
(<reason>); measuring`. The reason is one of `no stored entry`, `stored entry
unreadable`, `static key changed: <inputs>` (the inputs that moved, by name: the
store records one digest per static input, never their contents; an entry from
before that record only says the key changed), `stored manifest missing`,
`stored manifest bytes changed`, `stored plan unusable: <why>` or `live
admission refused: <why>`. `session.json` records the same outcome under
`plan_reuse` and the monotonic milliseconds of each start phase under
`phases_ms` (`yaml_inspect`, `runtime_identity`, `store_lookup`, `replay` or
`measure`, `helper_start`, `engine_start_to_ready`), written once per start.

The same entries are saved in `session.json` and returned by
`bc-serve status --run-dir DIR --json`. For helper-backed production starts,
inactive/degraded live placement emits a single-line `WARNING production
feature live_policy` with reason code and detail to the owner log. An
engine-only or otherwise not-applicable policy does not warn. The snapshot is
archived on a subsequent owner start so an old run cannot masquerade as a
new startup decision. Other optional features are reported without
implicitly being declared production-required. The native mapping and
file/status/warning paths have the card-free `bc_feature_state_test`
and `bc_session_test_part1` to `_part4` CTest targets.

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
no Python schema or shell parser; session and helper/engine process ownership
are native C++.

Build with `-DCMAKE_PREFIX_PATH=<installed radiance>` and
`-DRADIANCE_SRC=<radiance source>`; build `bc-serve bc_serving_test`
then run `ctest --output-on-failure`. Tests are CPU-only, including a
synthetic KFD/PCI/proc/sysfs tree. The resolver tests are engine-free: they select a
config through a C++ preset (`test_config_fixture.h`), generated from the shipped `engines/radiance/configs/NAME.yaml` by
`gen_fixture_presets.py` (a test fails when they drift); the production resolver reads a config only from its YAML.

Commands: `--detect [--json]`, `--schema`, `--validate <site> [profile]`, `--explain [--json]` (read-only). The manifest is written
by the session owner in-process (there is no `--write-manifest` command); the engine admits the plan its own start was given.
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
the engine starts from the native YAML and the owner's sealed admission (rad0070), the helpers from their typed settings; the shared `bc_serving_v2` library carries both documents.
The owner launches and verifies the measuring and serving process groups.

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
from a shell option parser. The existing `bc-serve --explain`
remains a lower-level, site/profile-oriented resolver inspection; no
compatibility flags are introduced. `bc_serving_test` execs the
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
A stored manifest is bound to the build that wrote it, its measurement is not:
when the replay checks refuse the manifest (a different engine or helper build)
the start reads the stored plan, rank demand and resident host memory back and
seals them again under the current build (the draft and measured writers run, no
helper or measuring engine does); the old build's identity and attempt are kept
as `plan_carried_from` in the selection and in `carried-from.json` beside the new
attempt. A stored measurement that cannot be carried (damaged, or no longer fits
the configuration) is measured afresh. The engine's own admission is the safety
net: an engine that fails to come up on a stored or carried plan removes the
entry so the next start measures. Without `--work-root` every start
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

A config that selects a different power profile than the recorded one is
refused. A stored lab manifest is refused: replay is production-only. There is no `--power-profile` flag.

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
and the native serving manifest process contract. **Difference from a shell
launcher:** no second syntax translates `--port`, `--heat`, `--budget`,
or other legacy flags into environment variables; the one resolver writes
the typed manifest (edition 3, sealed with the native YAML). `bc_typed_settings_test` executes the compiled `bc-serve
explain` against a synthetic KFD/
PCI/proc filesystem, real named configs, and producer-shaped route heat,
checking provenance, manifest fields and typed refusal cases. The test's
unmeasured manifest is not serving authorization. GPU parity remains S15.
