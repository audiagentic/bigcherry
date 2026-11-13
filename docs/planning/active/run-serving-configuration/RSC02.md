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

1. **Engine-only start (dense configs).** `bc-serve start CONFIG`: detect, resolve, reuse or write the manifest (plan
   cache and measuring start as today), start `radiance --manifest M --role engine`, wait for `/health`, run the
   canary, write a state file (config, manifest digest, PIDs, port, start time) under the run root. `stop` and
   `status` read the state file; stop signals only the PIDs it recorded and waits for their exit.
2. **Helpers.** Ordered helper start (helper 0 creates the shared segment), readiness, resident-memory reading for the
   plan, failure of any process stops the set; the measuring start with helpers up.
3. **Guards.** MTP replay guard, live placement report, heat recording; lab calibration stays a lab mode, refused
   by production configs.
4. **Power.** Apply the manifest's power plan as one transaction and restore it only after every child has exited
   (PWR09's transactional core; until it lands, power stays observe-only from `bc-serve`).
5. **Stored manifests.** `start --manifest FILE`, and reuse of a config's stored manifest (RSC12 step 5).
6. **Parity and cutover.** Same-session A/B of `bc-serve start` against the bash launcher on one build for one dense
   and one helper config (manifest digest identical; RHP12 gates). Then switch the callers (16 lab scripts, the release
   installer, tests) and delete the bash launcher and `lib/plan-check.sh` (RSC04).

## Detailed Solution & Technical Design

- Process control in C++ (`posix_spawn`, process groups, `waitpid`, signal forwarding), one state file per start;
  no daemon: `start` returns once the canary passes and the processes run on; `stop` finds them from the state file.
- A start never kills a process it did not start (PIDs from its own state file only), never applies power it cannot
  restore, and leaves the host as it found it after a failed start.
- `bc-serve` takes no lease: the container's `agent-lease` wraps the command in the lab (`agent-lease acquire gpu ...
  -- bc-serve start ...`), as it wraps the bash launcher today.
- Options that change settings (`--max-model-len`, `--kv-dtype`, `--budget`, ...) do not exist: a setting is a config
  value, a variant, or a `--set` experiment recorded in provenance.
- RSC11 decision 3 (a Python supervisor `bigcherry serve`) is replaced by this item (owner direction 2026-11-14): the
  `bigcherry` Python CLI may call `bc-serve start/stop/status` but does not supervise processes itself.

## Code Samples & Guidance



## Files



## Validation

- Card-free lifecycle tests with fake processes: ordered start, a helper or the engine failing during start, partial
  start cleaned up, repeated start/stop, stop of a dead set, a second start refused while one runs on the same port,
  inherited environment ignored, state file round trip, `--manifest` refused when stale.
- Hardware (gpu lease): one dense config (qwen27b TP3) and one helper config (flashnext-4gpu) start, serve, stop;
  same-session A/B against the bash launcher on one build passes the RHP12 gates with an identical manifest digest.

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
