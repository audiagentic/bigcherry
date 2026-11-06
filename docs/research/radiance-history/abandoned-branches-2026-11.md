# Abandoned branch disposition — November 2026

Nine branch tips were compared with main by **content** (including main's post-rewrite equivalents), not by the inflated commit counts caused by the August history rewrite. All source branches remain untouched. The repository archive is documentation and inert patches only, except the eight new **opt-in** layout entries copied into `engines/radiance/layouts.toml`; existing defaults were retained.

| Branch | What | Outcome | Why it stopped | Where archived |
|---|---|---|---|---|
| `exp/27b-fp8` | 27B FP8/W8A8+DFlash2 and RDNA2 tail budgets | Eight opt-in layouts retained; no branch-specific performance results | Unknown; experimental only | `experiments/radiance/27B-fp8-w8a8-layouts/` |
| `exp/rad47-xtx-rows` | gfx1100 MXFP4 tiled-vs-skinny/decode row priority | Already on main via #596 (+27% four streams, +37% six) | Superseded by merged equivalent | `experiments/radiance/RAD47-xtx-rows/` |
| `feat/rsc06-helper-manifest` | Helper startup from strict serving manifest | #527 closed; newer v1 contract merged via #633 | Original helper-config interface superseded | `experiments/radiance/RSC06-helper-manifest/` |
| `exp/router-splitk-with-cache-evict` | 1357 router split-K + 1358 eviction, 1330 QSA/ub1024 and A/B scripts | 1357/evidence/scripts on main; 1358 #107; 1330 ub1024 not adopted in #77 | Evaluated and retained/rejected through other PRs | `experiments/radiance/router-splitk-cache-evict/` |
| `feat/flashnext-4gpu-default` | Mixed-heat four-card launcher default | Shipped via #488; historical eight-session prefill +48% | Merged through equivalent PR | `experiments/radiance/RAD23-flashnext-4gpu-default/` |
| `feat/rad23-result-f16` | v21 fp16 helper results with row exponent | Unmerged novel v21 code preserved as inert patch | Replaced by newer protocol; specific validation/stop reason unknown | `experiments/radiance/RAD23-result-f16/` |
| `feat/rad23-result-f32` | v20 f32 helper partial results, combine and CPU bench | Unmerged novel implementation preserved as inert patch | Current v22+ combine design moved on; end-to-end trial unknown | `experiments/radiance/RAD23-result-f32/` |
| `rcs11-merge` | Pin HIP/ROCm compiler/runtime to selected SDK | Shipped via #524 | Already merged under main's rewritten history | `experiments/radiance/RCS11-rocm-sdk-selection/` |
| `redact-internal-refs` | Remove machine-bound references from source/config/docs | Main privacy cleanup #394/#399/#544 replaces legacy edits | Already applied/superseded; do not replay unsafe historical redaction diff | `experiments/radiance/internal-refs-redaction/` |

The two RAD23 patches are against **their recorded historical bases**, not clean patch sets against today's main. Apply the v20 `result-f32` patch first at `a20d1cb28`, and the v21 `result-f16` patch at `3732ffc59`. They are inert and need a protocol redesign, tests, KLD and throughput qualification to revive. The older router, default-layout, SDK and redaction branches predate a Git history rewrite; comparisons above rely on the branch-specific commit diffs, current files and the PR/ledger/plan record instead.

For counts, per-branch evidence and unresolved questions see each folder's README and `docs/research/radiance-history/EXPERIMENT-REGISTER.md`. No builds, GPU measurements, branch deletion, push or PR were performed.
