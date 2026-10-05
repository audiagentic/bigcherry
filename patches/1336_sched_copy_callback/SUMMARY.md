# 1336_sched_copy_callback

**Status:** untested
**Plan item:** MET01

Kind: upstream backport (llama.cpp PR #29943, not merged at the b11402 pin).

Moves the selective copying of host-resident MoE experts out of the scheduler's compute loop into a public copy
callback owned by `llama_context` (`sched_copy_experts`). Host weights are copied last so the callback can read the
split's other inputs. This is the seam MET01 requires before any expert cache: residency policy lives in user code.

The scheduler loop is edited in place rather than extracted as upstream does, so the patch composes with 1326.

BigCherry additions inside the callback:

- `BIGCHERRY_MOE_COPY=0` - observation-only control: count, then let the scheduler copy every host weight whole.
- `BIGCHERRY_MOE_COPY_DENSE_PCT` (default 90) - when nearly every expert of a layer is used (large prefill batches),
  one whole copy replaces hundreds of range copies.
- Counters at exit under `BIGCHERRY_PATCH_TRACE` (calls, bytes, experts used / total, dense shortcuts).

Without host-resident experts (the production tensor split keeps every expert in VRAM) the callback never fires.

Superseded when the pin reaches a llama.cpp release that contains #29943.

## Evidence

- Build on HIP, observation-only identity, selective identity with non-zero counters, multi-request integrity: pending.
