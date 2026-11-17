# Wide project scan and design review: power management in BigCherry and how to apply it through the serving manifest

Owner's question: "do we have power management in bigcherry - we built a lot of timings and settable fields but not sure how we captured
it - is it configurable via the manifest? Should we have power profiles in a configuration and select those?"

Repo: the main checkout (read via <host>): <repo> ; run artefacts under <work-root>/runs.
Read-only. Be adversarial: say where my picture below is wrong.

## What I found in a quick look (verify and extend)
- tools/bigcherry/cli/power_cmds.py + tools/bigcherry/hardware/gpu_power.py + hardware/power_setup.py: `bigcherry gpu-power status|apply|restore|grant|revoke`
  (PWR04): per-card OverDrive voltage offset, max sclk, board power cap; settings from config/environment.local.toml [host.*.devices.power]
  (schema documented in config/environment.example.toml, core/environment.py DevicePower); udev group grant; state file records the values to restore.
- tools/lab/radiance/launch-flashnext-best.sh `--gpu-power` (default off) applies the configured settings for a run and restores at stop.
- tools/bigcherry/cli/host_check.py warns when a card's power file is not writable and when RAPL is unreadable.
- Manifest: the only power field I found is spec.processes.helper_common runtime.power_save (R3_MOE_WORKER_POWER_SAVE, helper idles its GPU
  between requests). No GPU clock/cap/voltage, no CPU governor/EPP/C-state field.
- Plan items docs/planning/active/run-power-efficiency/PWR01-PWR06 all `pending` (whole-system baseline; board power cap sweep; CPU EPP/C-state;
  per-device settings + privileged helper + guarded controller; idle/load power of flashnext-4gpu; R9700 MES firmware trial).
- Current host state: all four cards on driver defaults (303/303/300/303 W caps), perf level auto, CPU governor powersave + EPP balance_performance
  (the engine's startup report flags the governor), the environment.local.toml has no power section.
- Radiance engine startup report notes: cpu-governor, swap-in-use, link-width notes (see <work-root>/flashnext/engine-measure.log).

## Scan I want (what was done, how, why)
1. Everything power-related in the repo: code, config, plans (PWR*, RAD*, RCS*, RHP*, MEN*, any "power"/"watt"/"joule"/"energy"/"RAPL"/"EPP"/"pstate"/"powercap"/
   "cstate"/"hwmon"/"amdgpu_top"/"rocm-smi" mention), tests, docs/reference, the ledger events, findings and run folders (runs/*power*, runs/pwr*, anything that
   measured watts or tokens per joule). For each: what it does, whether it is wired end to end, whether anything calls it, and whether results exist.
2. The measurement side: which timings and "settable fields" exist (telemetry, /metrics, engine startup report, helper stats, power sampling in benchmarks),
   how results are captured and where they are stored; is power/energy recorded in run outputs or the benchmark campaign schema at all?
3. The configuration side: how the existing device power settings flow (environment.py -> gpu_power -> launcher), what is validated (evidence field),
   what restores them, what happens on crash/kill (stale values left on cards?), privilege model.
4. Manifest and serving contract: the v1 contract is closed (docs/design/serving-manifest-v1, engines/radiance/components/serving/contract/README.md): how would power
   settings enter? Options: (a) a `spec.host`/`spec.power` section needing a contract-version decision; (b) named power PROFILES in config (like layouts.toml/archs.toml)
   selected by name in the manifest or launcher and resolved by bc-serve (bc-serve already resolves layout, objective, calibration); (c) keep them outside the manifest in the
   host environment but RECORD the applied state in the manifest provenance/run record. Compare, recommend one, give the schema.
5. Profiles: should there be power profiles (e.g. default, efficiency, quiet, max-throughput) keyed per card TYPE and per host, selected per layout/objective
   (service.objective already exists: "balanced"), validated per physical card with evidence, re-derived when hardware changes? How does that fit the repo's rules
   (no hand-picked constants: values must come from measured evidence; no legacy shims; layout-agnostic; fail-closed when unvalidated)?
6. CPU side: governor/EPP/C-states and who may change them (needs root; the container?), core pinning already exists (R3_MOE_WORKER_CPU_SET); what belongs in a profile.
7. Safety: undervolt/overclock can change numerics or stability; what gates are needed (output identity, KL, stability soak) before a setting is "validated"; how a start
   behaves when the profile is unvalidated for a card; interplay with the gpu lease (other agents share the cards!) and with other agents' runs (a setting applied by one run
   leaks into the next if restore fails).
8. How power state should be recorded in every benchmark result (so a result says which governor/EPP/caps it ran under), tying into the preflight proposed for live placement
   (RHP18: policy on/off/not-applicable recorded in results).

## Deliverable
(a) What exists, what is wired, what is dead or partial, with file:line. (b) A recommended design: where power profiles live, the schema, how the manifest/launcher selects and records them,
the validation gates and evidence rules, restore/failure behaviour. (c) A prioritised plan (PWR01-06 revised, what to measure first, what needs hardware, what needs root), sized.
(d) The single hardest design question, if any, written as a self-contained brief for a Fable-class session.
