# rad0070_managed_native_entry

**Status:** untested
**Plan item:** RSC02

Scope: The owner starts the stock native `--config FILE` entry and Fleet, and hands over only what it derived, as one sealed document (`--bc-admission-fd`).

## Summary

A managed start binds the owner's derived host sizes and rank devices onto the parsed YAML (an absent or auto value takes the derived one, an agreeing value stays, a disagreement refuses naming both), checks that the cards Fleet chose are the admitted ranks before the engine exists, and refuses reload (SIGHUP and HTTP), model admin and restart of an engine that stopped. Without the flag nothing changes.

## Outcome

Until the owner switches to this entry (RSC02-B) it sits beside the manifest entry, rad0009, which it requires for one include and one CMake line; the change that switches the owner archives rad0009.

## Detail

Policy lives in BigCherry's own component source (`components/serving/bc_serving_managed.cpp`, `v2/engine_admission.cpp`); this package only attaches it. Card-free test: `bc_native_managed_test`.

[Patch implementation](patch.py).
