---
id: RSC02
order: 2
plan: run-serving-configuration
state: pending
created-at: '2026-11-02T00:53:02.723581+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P0
work: M
---

# bc-serve start/stop/status: the only start-up path, in C++, from a named config or a stored manifest

## Description

Replace `tools/lab/radiance/launch-flashnext-best.sh` (749 lines of bash plus 292 in `lib/`) with start, stop and status
commands in `bc-serve`, the C++20 serving component that already detects the host, resolves the configuration,
measures and caches the memory plan and writes the manifest. Owner direction 2026-11-14: one language for the start
path, radiance's (C++); the launcher is simple to read and run; it starts a named config (RSC12) or a stored manifest.

```
bc-serve start qwen27b/tp3-int8-556           # resolve (or reuse) the config's manifest, start, health, canary
bc-serve start --manifest FILE                 # rerun exactly a stored manifest; refused when its key no longer holds
bc-serve stop                                  # stop everything this start owns, then restore power
bc-serve status                                # config, manifest digest, processes, port, power state
bc-serve explain qwen27b/tp3-int8-556          # effective values and where each came from; starts nothing
```

What the bash launcher does today and the command keeps (inventory from origin/main, 2026-11-14): host site file and
SDK check; plan check with the measuring start (helpers up, engine `--plan-memory`) and the host-pool round; helpers
started in order and their resident memory read; engine started from the manifest only; `/health` wait; start canary
(deterministic completion, configurable prompt and expectation); MTP replay guard; GPU power apply and restore; gfxoff
floor; route-heat file and live heat recording; live placement policy report; lab calibration mode; cleanup of a
partial start; `--release TAG` (start from a verified release). Every one becomes a config field, a host-file field or a
step of `start`/`stop`, never a launcher option that rewrites settings.

## Steps

1. **Session owner, engine-only (dense configs, today's layouts).** `bc-serve serve|start|stop|status`: resolve, plan
   check and measuring start, engine from the manifest, `/health`, canary, control socket, ordered stop. Card-free tests
   ported from the Python supervisor's fakes first; then one dense layout on hardware.
2. **Helpers.** Ordered helper start and readiness, resource derivation in C++, measuring start with helpers up, any
   child's failure stops the set.
3. **Guards and lab modes.** MTP replay guard, live placement report, heat recording, lab calibration (refused by
   production configs).
4. **Power.** The manifest's power plan applied as one transaction by the owner, restored after verified exit (PWR09).
5. **Configs and stored manifests** from RSC12: `serve CONFIG` on the new files; `start --manifest FILE` replays a
   stored manifest without rewriting it (RSC12 step 5).
6. **Acceptance and retirement in RSC11's order:** the coordinated v2 switch (S13) stays one change; hardware and release
   acceptance (S15): same-session A/B against the bash launcher on one build, one dense and one helper config, RHP12
   gates, equal semantic execution projection (RSC12); retirement (S16): switch the callers (16 lab scripts, the release
   installer, tests) and delete `launch-flashnext-best.sh` and `lib/plan-check.sh` (RSC04).

## Detailed Solution & Technical Design

**A session owner process (review 2026-11-14, P0).** `bc-serve serve CONFIG` is a foreground C++ process that owns
the whole serving session: it starts the children in their own process groups, reaps them (`waitpid`), watches health,
holds the power journal, and on any child's exit or on a stop request stops the rest in order and restores power only
after every child has exited. `bc-serve start CONFIG` is `serve` detached (`setsid`), returning once the canary passes;
`stop` and `status` talk to the owner over a control socket in the session's run directory (never by signalling a
recorded PID, which can be reused). In the lab the GPU lease wraps the owner (`agent-lease acquire gpu ... -- bc-serve
serve CONFIG`), so it is held for the session's lifetime, not just the start.

- Reuse what exists in C++: the SDK check (`sdk_check.cpp`, already called by `bc-serve`), the resolver, the plan cache,
  the manifest writer and reader, the host ledger and the power probes.
- The Python supervisor (`tools/bigcherry/serving/supervisor.py`, RSC11 S6/S11/S12) is the behavioural reference: its
  fake-process, lease and power-transaction tests are ported as the C++ owner's acceptance tests; it is not on the start
  path.
- Options that change settings (`--max-model-len`, `--kv-dtype`, `--budget`, ...) do not exist: a setting is a config
  value, a variant, or a `--set` experiment recorded in provenance. `--ngram-device` is not carried over (manifest
  starts refuse it today; it needs a typed manifest field first).
- Behaviour to keep, with the bash references (from the GPT review, req_rsc02-config-launcher_957d54878bcc):
  release mode (verified tagged bundle; `--release` and `--build` exclusive); SDK and ROCm environment refusal;
  a launch/stop lock (no concurrent measuring, plan storage or stop); stop order engine SIGINT then helpers, bounded
  waits then SIGKILL, never success while a process survives, power journal kept on a failed restore; binary and kernel
  library checks; BDF to HIP ordinal mapping, rank order, per-process GPU visibility; RSC03 resource derivation for
  helper starts (today `bigcherry resources derive-launch`, moved into C++); the measuring start (helpers up first,
  helper RssAnon + RssShmem recorded once, per-rank demand, exit 10 cache miss and exit 11 derived pool kept apart, no
  silent fallback); heat file and lab experiment hashed, histogram path and period, live policy fail-closed; ordered
  helper readiness from the fresh run, shared segment before the engine, `/health`, canary, MTP replay guard (depth > 1);
  power applied before measuring and serving, restored after verified exit; gfxoff floor and loaded-idle check; lab
  calibration (helper layouts only, two distinct prompt files, ABBA, strict ingestion); helper-free and helper layouts'
  different allowed overrides.

## Code Samples & Guidance



## Files



## Validation

- Card-free lifecycle tests with fake processes (ported from the Python supervisor's): ordered start; a helper or the
  engine failing during start and while serving; partial start cleaned up; repeated start/stop; stop of a dead set; a
  second start refused while one runs; stop through the control socket only; power restored only after every child
  exited, journal kept when restore fails; inherited environment ignored; `--manifest` refused when stale, naming why.
- Hardware (gpu lease held by the owner for the session): one dense config (qwen27b TP3) and one helper config
  (flashnext-4gpu) start, serve, survive a killed helper (set stopped, power restored), stop; same-session A/B against the
  bash launcher on one build passes the RHP12 gates.

## Effort & Risk



## Standards



## Acceptance Criteria

- `bc-serve start CONFIG` starts every configured model, dense and helper layouts alike; `bc-serve stop` stops it
  and restores the host's power state.
- `bc-serve start --manifest FILE` reruns a stored manifest exactly or refuses, naming the input that changed.
- No process receives configuration through flags other than `--manifest`/`--role`, or through environment variables.
- The bash launcher and `tools/lab/radiance/lib/plan-check.sh` are deleted; no caller references them.
- `bc-serve explain` (dry run) prints the manifest and the process plan without starting anything.

## Notes

Depends on RSC01. Source: GPT configuration assessment 2026-11-02.

Depends on RSC01 (loader), RSC06 (helpers read the manifest) and RSC07 (engine reads the manifest). Owner decision 2026-11-02 (recorded on RSC01): no command-line converter and no fallback path.



BACKLOG TRIAGE 2026-11-04 absorbed items (superseded into this one; their notes keep the full history): MEN14: Remaining cutover (one workflow, lab launcher retired) is RSC02/RSC04 (bc-serve as the only start-up path); both exist and are open.; MEN18: `bigcherry remote run|status|wait|list` exist (tools/bigcherry cli); leftover launch utilities belong to RSC02.



CORRECTION 2026-11-04: the earlier note that MEN14, MEN18 were absorbed into this item is withdrawn. An independent revalidation found each either still has open scope of its own (reopened) or was delivered outright. This item's scope is unchanged.

OWNER DIRECTION 2026-11-14: rescoped. The command is `bc-serve start|stop|status|explain` in C++ (radiance's language),
not `bigcherry serve` in Python; it starts a named config from RSC12 (one readable file per model, named variants) or a
stored manifest. Depends on RSC12 for configs; steps 1-4 can start on today's layouts. GPT is the backup reviewer.

## Change Log

- 2026-11-02T00:53:02.723581+00:00 (created-by): Created by agent

## Ledger-events

- chg_20261102_005338_planning-one-serve-command_5259
- 2026-11-02T00:53:48.446911+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-02T03:51:40.629364+00:00 (updated-by): Updated: section:title, section:steps, section:acceptance_criteria, section:notes
- chg_20261102_035356_serving-start-up-plan-one-str_4985
- 2026-11-02T03:54:04.045012+00:00 (updated-by): Updated: section:ledger-events
- 2026-11-04T03:47:55.795874+00:00 (updated-by): Updated: section:notes
- 2026-11-04T04:45:47.730785+00:00 (updated-by): Updated: section:notes


## Update 2026-11-10 (GPT power manifest review, req_main_eb983900e21b)

Acceptance gate added: the Python power transaction (PWR09) and its receipt lifecycle are part of supervisor acceptance (apply before the first measuring or serving process, restore after verified shutdown). bc-serve starts no processes today and the shell launcher remains the supervisor (it also does readiness waits, the canary, MTP replay verification, cleanup); recommended: a thin Python bigcherry serve calling the C++ resolver/validator, not a second resolver. The launcher stays until RSC02 reaches equivalence (RSC04).
