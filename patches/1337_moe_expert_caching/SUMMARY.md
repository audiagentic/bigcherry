# 1337_moe_expert_caching

**Status:** untested
**Plan item:** MET01

Kind: upstream backport (llama.cpp PR #29887, commit 6b7b03aab, not merged at the b11402 pin). Requires 1336.

With `--moe-cache-mib N` (`LLAMA_ARG_MOE_CACHE_MIB`), MUL_MAT_ID ops whose expert weights are kept in host memory
(`--n-cpu-moe`) run on the GPU against a persistent device buffer: the experts a batch selects are uploaded on a miss
into least-recently-used cache slots, and the op reads a remapped ids tensor naming the slots. Without the flag
nothing changes. Upstream limits apply: one device, no pipeline parallelism.

The patch is the upstream commit hunk for hunk, with the seven hunks that overlap 1336 anchored on 1336's form, and
creates `src/llama-moe-cache.cpp` and `.h`.

The published PR does not sit on the #29943 copy callback: it adds its own scheduler hooks and rewrites the graph at
split time, which a copy callback cannot do.

Superseded when the pin reaches a llama.cpp release that contains #29887.

## Evidence

- Build on HIP; equal-VRAM lanes on gfx1201 (0 vs 4096 MiB with `--n-cpu-moe 41`); multi-request integrity: pending.

- build moe-cache-profile compiled clean and is byte-identical in output with the flag off (smoke md5 equals production).
- Real host-expert 0 vs 4096 MiB vs profiled-cache qualification at b11474 is still pending; use `tools/lab/flash-next/queue-moe-cache.sh`.
