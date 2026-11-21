# LTX-2.5 IQ8R — actual device kernel checks, 2026-11-22

**Scope:** Three independent real GPU checks of the native LTX kernel vocabulary and actual IQ8R checkpoint projection kernels, using an exclusive `gpu` lease. This is **not LTX TP3 execution or parity**. The C06 source-backed attention graph is still opt-in diagnostic-only and LTX serving still refuses TP3.

## Hardware and library selection

| GPU | Architecture | Native library | Checkpoint GEMM device cases passed | Fail | Diverged |
|---|---|---|---:|---:|---:|
| RX 7900 XTX #0 | gfx1100 | `libr3.so` | 320 / 320 | 0 | 0 |
| RX 7900 XTX #1 | gfx1100 | `libr3.so` | 320 / 320 | 0 | 0 |
| R9700 | gfx1201 | `libr4d.so` | 321 / 321 | 0 | 0 |
| **Total** | | | **961 / 961** | **0** | **0** |

Used the released Radiance 2.2 SDK's `rad-kbench` against the **real LTX-2.5 IQ8R RAD2 model file**, `--op gemm_nt_q` and `libref` oracle, reading checkpoint weight payloads and using each device's native kernel. The bounded declaration needed the LTX model's minimum max-token capacity of **33,833**; a preliminary run with four was correctly rejected. The valid runs constrained the per-case allocation to **512 MiB**, `--max-cases 1`, and `--max-ctx 4096`.

Worst relative L2 error on each card: **2.848e-4**. R9700: 48 memory skips; each XTX: 49 memory skips. Every card reported 20,863 case instances capped by the case limit, zero guarded-buffer overruns. These are **scoped** checks, not an exhaustive all-weight/full-shape or end-to-end comparison.

## Executed LTX GPU kernel numerical tests

Seven distinct selftest case selectors **passed on each of the three cards** (**21/21 invocations**, zero skips/failures):

- `rope_split`: video three-axis/full-dim 4096 and audio one-axis/full-dim 2048, with global head offset, neighbouring columns preserved; video relative L2 **2.41e-5**, audio **4.16e-9**.
- `head_gate`: 32 heads of width 128, nontrivial logit-row pitch.
- `had_quant_act_iq8r`: real device group-256 rotated activation quantization numerical checks.
- `gemm_iq8r_nt_m16`: group-256 IQ8R GEMM numerical suite, including M=58/N=2048/K=10240. This selector shares its 35 individual assertions with the preceding IQ8R selector; **do not double-count** them as independent tests.
- `rmsnorm_bf16`: reference numerical and row-count invariance.
- `attn_dense_h128_bf16`: noncausal video dense attention, 3001 rows / 2 heads and 1333 rows / 3 heads, relative L2 about **0.003**, within 0.02 tolerance.
- `attn_dense_h64_bf16`: audio dense attention, including noncausal four-sequence 1797-row and short causal-window variants; worst listed relative L2 about **0.00131**, within 0.02 tolerance.

For gfx1201, this used the installed native `r4d_selftest` executable and `libr4d.so`. The native selftest source's device phase hardcodes gfx12, so gfx1100 used a **work-root, lab-only copy** of the unchanged numeric selftest source with *one edit to the device-architecture filter*, built as a separate executable against the **installed `libr3.so`**. No kernel library, source recipe, worktree C06 architecture, or production binary was modified for these runs. The small `libr3d.so` plugin is not the comprehensive gfx1100 library; the installed `libr3.so` has the required IQ8R, RoPE, head-gate, dense-attention and exact-collective operator rows.

Only the `rad-kbench` projections used the real checkpoint payloads. The seven selftest selectors used their own deterministic numerical inputs, not checkpoint activations.

## Raw evidence and reproducibility

The same-session work-root run directory `ltx25-tp3-c06-gpu` contains the three `model-device*-gemm_nt_q.md` Markdown reports, matching `.log` and `.json` receipts, independent `device2-*.log` and `device{0,1}-libr3-*.log` native selftest logs, and local isolated launch/build scripts. These outputs are not added to the repository because they are machine-specific, large and/or temporary.

The native GPU process inventory was checked before and after execution. Tests ran under an exclusive GPU lease, which was successfully released after the final GPU process exited.

## Boundaries and next gate

**Not done:** a C06 `Attn` model-weight issue run; same-request multi-rank Q/K RMS and output reductions; cross-card barrier/collective and recovery; TP1-vs-TP3 numerical output comparison; full 48-block DiT or video generation. The current native C06 tests **only declare/finalize the graph**. Kernel tests confirm that the prerequisite device vocabulary and actual IQ8R projections work independently on all three target cards, not that those kernels have been wired into an executed TP3 request.

**Next:** introduce one real C06-issued rank-local self-attention graph using checkpoint weights and compare it against TP1. Then coordinate the existing native exact TP3 collective across gfx1100/gfx1100/gfx1201 with the actual two legal head layouts and validate alignment, cancellation, error handling and numeric parity before admitting TP3 serving.
