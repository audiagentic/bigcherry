---
id: BCOP113
order: 113
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-10T10:05:08+00:00'
created-by: agent
priority: P2
work: S
---

# PNRO04: reconcile validated BF16 GDN and bound unsupported-ratio admission

## Discovery / change

1253 is **validated and production-selected**, not predicate-only/untested. Historical b11126 four-session gfx1100/gfx1201 evidence is real; old PNRO04 and README statements conflicted with `patch.toml` (`requires=[]`, `conflicts=[1221]`). The actual K=1 BF16 dispatcher admits ratios outside the KKT wrapper set `{1,2,3,4,6,8}`, whose default is `GGML_ABORT`. BF16 `bits+0x8000` differs from ties-to-even on exact halfway inputs. These are source/host findings, **not reproduced GPU failures**. Updated PNRO04 and 1253 README/SUMMARY; no patch, recipe or evidence changes.

## Ownership / existing work / exclusions

PNRO04/1253 owns K=1 BF16 GDN; 1221 is an incompatible FP32 alternative. PNRO05/1254 owns untested K>1 MTP state/snapshot work; no change underneath that owner. Upstream llama.cpp #29353 (open, HIP gfx115x-only) is a reference, not an RDNA3 gfx1100/RDNA4 replacement. Excluded active Radiance gfx1100 FP8/MXFP4/KL and serving, Flash-Next 1330/1334/1347, router 1357, Meta 1355/1356/1358, and DFlash/MTP acceptance work from the preceding 12 hours. Previous BCOP103–112 slices are not repeated. No independent 1253/PNRO04 implementation, PR, benchmark, or queued test in the preceding 12 hours was found.

## Unresolved action / terminal gate

First run host selector ratio/architecture/flag fixture and pristine b11474 patch-composition/build check; **reject any expansion** that aborts on unsupported ratios or alters state publication. Reuse existing qualification infrastructure, no competing hardware queue. Only if an actual supported workload hits the route, measure b11474 on/off KKT/scan critical path, scratch/pool/graph lifetimes, and full-vocabulary/greedy/KLD + state parity with four-session paired hardware controls. Close without further code if no >=3% theoretical E2E opportunity; promote a new default-off change only on >=3% CI95-low E2E with <=1% controls and no correctness regression. Historical validated 1253 remains unchanged.

## Evidence

`config/recipes.toml`, 1253 `patch.toml`/`patch.py`/`evidence/validation.json`, pinned b11474 `gated_delta_net.cu`; upstream llama.cpp PRs #29353/#30087/#30207, vLLM #56307, SGLang #39873. Disposable integer host fixture: 32,512 halfway cases, 16,256 BF16-RNE differences; synthetic GQA ratios 5 and 16 reach unsupported switch. No C++/HIP compile, GPU execution, new benchmark, or current-pin validation in this run. Technical design and acceptance gates reside in PNRO04.
