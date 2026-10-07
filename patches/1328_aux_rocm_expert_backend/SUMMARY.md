# 1328_aux_rocm_expert_backend

**Status:** untested
**Plan item:** MET05

## What it does

Adds an opt-in `BIGCHERRY_EXPERT_AUX_DEVICE=<device>` path for Qwen4Exp tensor-split target contexts. The named GPU is registered as an ordinary scheduler backend without being added to the Meta tensor group. Routed-expert tensors placed there with `-ot` execute there, with Meta/aux activations copied through a scheduler-owned pinned host buffer.

For a fully auxiliary routed-expert layer, the routed MoE result is treated as a full mirrored value. If the shared-expert branch is still PARTIAL at the merge, Meta reduces it once; if it is already MIRRORED, both inputs are complete replicas and the merge is a plain per-device add. The auxiliary result is never AllReduced.

## Safety

Off by default. The path fails closed if the auxiliary device is absent, ambiguous, non-GPU, already in the Meta group, if a selected routed-expert layer is only partially on the auxiliary device, if router/shared-expert tensors are moved there, or if a Meta/aux crossing is not MIRRORED.

## Why

`-ot ...=ROCm3` can allocate target weights in an ordinary ROCm3 buffer, but the target scheduler previously knew only Meta + CPU and aborted during graph reserve. This package makes ROCm3 schedulable without adding it to RCCL/cpu-root AllReduce membership and preserves exact routed/shared reduction semantics.

## b11474 diagnosis

The first hardware sweep after the enum-tag compile fix aborted at load for every aux-offload layout because the marked merge observed `MIRRORED + MIRRORED`, while 1328 only accepted `MIRRORED + PARTIAL`. At b11474, Meta's split-state contract uses `MIRRORED` for a complete replica; its synchronized matmul state can therefore make the shared branch MIRRORED before buffer initialization. In that state an AllReduce would incorrectly sum duplicate complete values. 1328 now accepts both `MIRRORED + PARTIAL` and `MIRRORED + MIRRORED`, returning MIRRORED in either case, and still aborts on every other marked combination.

## Hardware result and decode review (Brutus 2026-10-08)

Flash-Next, 245760 ctx, 79722-token fill, auxiliary routed-expert layers on ROCm3 (6900 XT):

```text
ub512:  L0 1117.7 t/s (decode 65.3 t/s, accepted 336/525; VRAM GiB 22.7 23.1 31.1 4.1)
        L2 1058.5 (22.3 22.6 29.7 6.4) | L4 1001.8 (21.8 22.1 28.4 8.7) | L6 964.1 (21.3 21.6 26.8 11.3)
ub1024: L0 OOM (cudaMalloc 100 MiB on device 1) | L2 1158.8 (23.4 23.7 30.5 7.1) | L4 1136.7 | L6 1086.6
ub2048: all arms OOM (3958.87 MiB on device 0), including L0
```

At equal ubatch the offload costs about 5% prefill per two layers, but L2 frees enough VRAM for ub1024. Every L2/L4/L6 arm then stopped at the first generation step while L0 decoded normally.

Follow-up after d2c8fd0c and bf25b05a showed the same failure. A 10102-token L2/ub512 diagnostic ended after `created context checkpoint 2 of 32 (pos_min = 10097 ...)`; the server process then disappeared without an error, assert, abort trace, or ggml_abort. The only 1328 runtime line was a 20480-byte Meta/ROCm3 scheduler staging copy; neither marked merge trace fired. The longer 79722-token rerun measured L2/ub1024 at 1178 t/s and L2/ub512 at 1059 t/s; no-offload ub512 was 1087-1118 t/s and generated normally.

Focused review found one concrete 1328 correctness defect: the generic scheduler copy hook identified the auxiliary endpoint only by the process-global `BIGCHERRY_EXPERT_AUX_DEVICE` device name. That is not context/scheduler ownership and can make an unrelated scheduler using the same ordinary GPU enter 1328's Meta/aux staging path. 1328 now stores the exact auxiliary backend pointer in the target DEFAULT scheduler and rebinds it after every scheduler reserve; the copy hook requires pointer identity. This is a required scoping fix, but static review does not prove that it caused the observed first-generation process death.

The reviewed decode-specific neighbors do not expose another concrete fault: 1340 handles current-graph replan/binding for the smaller graph; 1341 makes synchronous/asynchronous Meta MIRRORED transfers honor subset active masks; 1295/1327 and 1326 do not directly dereference the auxiliary routed tensor; NextN/embedding extraction resolves the scheduler backend and copies into host output. No source-level defect in those paths can be tied confidently to the silent death from the available log.

With `BIGCHERRY_PATCH_TRACE=1`, 1328 emits WARN-level `BIGCHERRY_PATCH_HIT ... hook=<name>` diagnostics at target registration/scheduler scope, Qwen4Exp layer entry and merge marking, scheduler assignment/copy, and marked split execution. Trace is otherwise silent so qualification throughput is not perturbed. The lab runner records `SERVER_EXIT status=<n> signal=<name>` so a SIGSEGV/SIGKILL is preserved in sweep output.

The exact process-death root cause remains unresolved at b11474. Do not treat the scheduler-scope change as a validated decode fix; promotion remains blocked.

## Upstream

Local BigCherry feature against llama.cpp pin b11474 (`b9acf138a1e2`).
