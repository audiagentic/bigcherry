# rad0070_managed_native_entry

**Status:** validated
**Plan item:** RSC02

Scope: The owner starts the stock native `--config FILE` entry and Fleet, and hands over only what it derived, as one sealed document (`--bc-admission-fd`).

## Summary

A managed start binds the owner's derived host sizes and rank devices onto the parsed YAML (an absent or auto value takes the derived one, an agreeing value stays, a disagreement refuses naming both), checks that the cards Fleet chose are the admitted ranks before the engine exists, and refuses reload (SIGHUP and HTTP), model admin and restart of an engine that stopped. Without the flag nothing changes.

## Outcome

Validated on the shared acceptance of the native managed start (docs/evidence/2026-11-26-rsc02-b-acceptance). The owner starts every engine through it. It owns the engine's entry (the stock `--config` YAML entry run after the owner's flags are taken), the shared serving library link and the engine-side sources; the manifest entry it replaced (rad0009) is archived.

## Detail

Policy lives in BigCherry's own component source (`components/serving/bc_serving_managed.cpp`, `v2/engine_admission.cpp`); this package only attaches it. Card-free test: `bc_native_managed_test`.

Attachment points:

The attachment points of the managed native entry. See SUMMARY.md and `components/serving/bc_serving_managed.h`.

| Anchor | Why there |
|---|---|
| `core/main.cpp` before `if (check)` | after the YAML parse and the unsupported-key refusal, before Fleet or the offline Engine read the Config |
| `core/fleet/fleet.cpp` before `cfg.devices = chosen;` | the one place Fleet fixes a load's devices, before the Engine is constructed |
| `Fleet::reload()` | both SIGHUP and `/admin/reload` reach it |
| `POST /admin/models/:name/:action`, `POST /admin/reload` | the model-mutating routes answer 403 |
| `Fleet::reap()` and the return of `Fleet::run()` | an engine that stops ends the process with a failing exit status; nothing demands it back |

The state is process-global and set once (`bc_serving_managed()`), by design.

[Patch implementation](patch.py).
