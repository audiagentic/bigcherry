---
{}
---

# Tool Disposition — current registry

## Description

This is the current control-plane registry for in-scope tooling. The
registry had 385 rows at TR00 close-out and now records maintained tooling,
current transitional labs, and explicit ownership decisions for supported
compatibility or machine-local paths. Retired lab implementations and moved
run outputs are removed from this live table when they leave `tools/lab/`;
their provenance remains in plan/evidence records and Git history.

This is the maintained disposition authority consumed by
`tools/bigcherry/check.py`. It is current-state inventory: every table row names
one tracked file in this checkout. Historical, machine-local, moved, or deleted
paths belong in plan/evidence records and Git history rather than as tombstones
in this live table.

The immutable 383-row implementation-start baseline is preserved in the
tracked TR00 evidence bundle at
[`docs/evidence/tooling-rationalisation/TR00/`](../../evidence/tooling-rationalisation/TR00/).
That bundle is historical evidence and is not edited to change current
ownership.

## Steps

## Detailed Solution & Technical Design

## Code Samples & Guidance

## Files

## Validation

## Effort & Risk

## Standards

## Acceptance Criteria

## Notes

## Rules

- Every in-scope row has exactly one path, one disposition, and one rationale.
- The current registry is the sole live disposition table; do not create a
  second registry in a plan, archive, or new tooling module.
- Every disposition row must name a tracked file in the current checkout; remove or replace the row in the same change that removes or moves the file.
- A move, retirement, graduation, or new lab file requires updating this
  registry, the owning plan/evidence references, and the focused hygiene or
  boundary checks in the same change.
- No implementation was moved or deleted merely to produce the TR00 baseline.
  Historical TR00 rows remain frozen under `docs/evidence/`; current changes
  must update this table without rewriting that evidence.

| Path | Disposition | Intended owner / rationale |
| --- | --- | --- |
| `engines/llamacpp/patches/_template/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0100_cmake_options/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0200_dispatch_hook/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0300_mmq_forced_j/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0400_mmvf_forced_block/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0500_mmf_forced_nwarps/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0600_mmvq_geometry/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0650_mmvq_native_variant/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0700_coverage_counters/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0800_server_shutdown_endpoint/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0810_replay_hit_diagnostics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0820_measurement_signature_shapes/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0830_split_reduce_telemetry/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/0900_pool_workspace_metrics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1000_rdna4_mmq_q2k_q6k_fix/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1002_hip_unsafe_math_opt_in/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1003_quantized_cpy_thread_block_fix/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1004_rms_norm_mul_rope_fusion/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1005_prompt_cache_checkpoint_selection/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1100_hi70_direct_op_evidence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1200_rd19_single_gpu_meta_bypass/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1201_rd20_attn_gate_tp_split/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1202_rd04_bf16_flash_attn_tile/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1204_rd08_q6k_mmvq_vdr2/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1205_rd12_paired_mmvq_dual_output/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1206_rd13_mul_mat_add_view_fusion/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1207_rd17_moe_topk_down_fold/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1208_rd21_gfx1151_mmvq_nwarps_table/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1209_rd22_integrated_gpu_host_buffer_backout/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1210_rd26_bitidentical_decode_verify_standalone/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1215_rd394041_amd_stream_moe_overlap/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1216_rd43_concurrent_join_fusion_guard/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1217_rd44_graph_opt_default_rdna35/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1221_rd50_gdn_chunked_recurrence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1223_hi67_machine_readable_correctness_metrics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1224_hi18_reduce_correctness_probe/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1230_hip_autotune_inspect/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1231_hi14_graph_capture_lifecycle_evidence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1232_hi81_windows_cxx_hipcc_flags_reach_compile/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1233_rd73_stable_graph_cache_key/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1234_rd58_pin_state_buffer_multigpu_restore/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1236_hi105_deterministic_mul_mat_id_ids/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `engines/llamacpp/patches/1237_rd30_moe_mmq_compact_grid/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `tools/bigcherry/__init__.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/__main__.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/analysis/candidate_report.py` | **KEEP** | Maintained analysis implementation; invoke with `PYTHONPATH=tools python -m bigcherry.analysis.candidate_report`. |
| `tools/bigcherry/check.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/device_state_validate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/doctor.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/e2e_smoke_campaign.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/e2e_smoke_report.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/generalise.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/graph_lifecycle_evidence.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/hi16_forced_native_parity.py` | **GRADUATE** | Generic native-versus-forced parity mechanics; extract behind a maintained correctness API without retaining HI16 naming. |
| `tools/bigcherry/hi18_run_corpus.py` | **TRANSITIONAL** | HI18-specific corpus runner; separate reusable reduction correctness from plan-specific corpus data before graduation. |
| `tools/bigcherry/hi80_generate_correctness_evidence.py` | **GRADUATE** | Generic correctness-evidence generation mechanics; extract behind the maintained evidence API before retiring historical entrypoint. |
| `tools/bigcherry/identity_separation.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/inventory.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/lifecycle.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/moe_hostile_routing_sweep.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/moe_routing_gen.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/parity_loaders.py` | **TRANSITIONAL** | Historical parity/cutover helper; retain until permanent invariant ownership and zero-caller proof. |
| `tools/bigcherry/parity.py` | **TRANSITIONAL** | Historical parity/cutover helper; retain until permanent invariant ownership and zero-caller proof. |
| `tools/bigcherry/patcher.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pin_transition.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pool_protocol.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/rank_replay.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/re14_real_run.py` | **TRANSITIONAL** | Historical acceptance harness; preserve campaign/artifact invariants before retirement. |
| `tools/bigcherry/re15_acceptance_run.py` | **TRANSITIONAL** | Historical acceptance harness; preserve campaign/artifact invariants before retirement. |
| `tools/bigcherry/re15_tamper_evidence.py` | **TRANSITIONAL** | Historical tamper/evidence harness; retain until integrity checks have a permanent owner. |
| `tools/bigcherry/recipes.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/release_validate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/replay_build_audit.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/replay_cache.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/rocprof.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/transform_loader.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/transform_records.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/validate_rd_patches.py` | **TRANSITIONAL** | Historical RD validator; generic validation remains authoritative. |
| `tools/bigcherry/vk_autotune_types.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/lab/ar-accuracy/gates.py` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/kld.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-ar-accuracy.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-kld-adaptive.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-kld-decode.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/bump-validation/run_bump_validation.py` | **KEEP** | RHA12: standing bump-validation matrix (PIN_BUMP.md step 5/6) -- builds fresh at the current pin and launches the real production runtime-profiles across every real GPU individually plus the real dual-XTX multi-GPU topology; run on every future bump, not a one-shot experiment. |
| `tools/lab/bump-validation/smoke_worker.py` | **KEEP** | RHA12: the real per-cell delegate_argv worker run_bump_validation.py's runtime-matrix cells launch -- reuses ServerRunner to actually start each server, wait for /health, send one real completion, and shut down cleanly; part of the same standing bump-validation tool, not a one-shot experiment. |
| `tools/lab/cross-engine/probe-openai.py` | **TRANSITIONAL** | `cross-engine` lab topic file (see `tools/lab/cross-engine/README.md`); experiment-only, disposed per that README. |
| `tools/lab/cross-engine/probes-compare.py` | **TRANSITIONAL** | `cross-engine` lab topic file (see `tools/lab/cross-engine/README.md`); experiment-only, disposed per that README. |
| `tools/lab/cross-engine/run-quality-27b.sh` | **TRANSITIONAL** | `cross-engine` lab topic file (see `tools/lab/cross-engine/README.md`); experiment-only, disposed per that README. |
| `tools/lab/default-on/smoke-0910.sh` | **TRANSITIONAL** | `default-on` lab topic file (see `tools/lab/default-on/README.md`); experiment-only, disposed per that README. |
| `tools/lab/default-on/xmodel-ab.sh` | **TRANSITIONAL** | `default-on` lab topic file (see `tools/lab/default-on/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/dflash-accept-iso.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/dflash-depth.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/drafter-files.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/fetch-drafters.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/probe-27b.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-cpuroot.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-detail.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-dflash.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-q4.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/_anchor_diff.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/abba-depths.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ahead-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ar-boundary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ar-segment.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ar-trace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/attn-maxctx.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/balance-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/census-run.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chain-serial.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-nomtp-diag.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-nomtp-diag2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-nomtp.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-prof.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/combined-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/cross-model-rel.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ctx-fit-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/deploy-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/determinism.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft-quant.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft-timing-summary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft27b-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft27b-single.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/expert-6900-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/expert-offload-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/expert-place.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/fa-sparse-backend-test.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/fit-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/fit-sweep-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/fixed-ts-ctx-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/flash-fidelity.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/flash-prefill-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/flash-prefill-env-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/flash-probs-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-numeric.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-ref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-retest.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-v2-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gemma-iso.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gpu-usage-sampler.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/graph-memlog.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/graphs-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/indexer-backend-test.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/kpool-ab-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/kpool-parity.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/layout-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/long-ctx-fit.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/long-ctx-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/mask-ref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/mask-ref2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-f16k.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/meta-timing-summary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/mixed-batch-test.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/mmid-range-test.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/moe-copy-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/native-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/page-cache-fraction.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/prefill-kernel-table.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/prefill-provider-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/probes-compare.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/probs-compare.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/prod27b-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/profile-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/q81-trace-run.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1301-widths.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1302-deepfill.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1303-deepfill.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1326.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1327.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1330.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-1330e.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-27b-pairs.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ahead-ctl.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ahead-gate.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ahead-pmin.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ahead-screen.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-alloc-top.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-apitrace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-arena-content.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-attn-maxctx-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-attn-maxctx.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-balance-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-bump-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-bump-census.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-bump-phases.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-bump-smoke.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-bump-v5.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk-confirm.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk-prof.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk-ub1024.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk3.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk7-diag.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk7-diag2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk7.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk8.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk9.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-combined-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ctx-fit-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-default-on.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-defaults.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-deploy-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism3.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism4.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism5.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-dflash-depth.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-draft-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-draft-host.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-draft-quant.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-env-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-expert-6900.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-expert-cpu.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fa-default.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fit-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fit-probe2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fit-sweep-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-ar-trace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-ar.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-cpuroot.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fmtp-calib.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fmtp-calib2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fusion-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gate0.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric-2gpu.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric-f16.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-ref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-v2-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-v2-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-graphs-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-indexer-tile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-ab-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-parity-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-parity.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-decode.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-fit.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-perf.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maskref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maskref2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-f16k-vram.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-f16k.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-search-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-search.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-meta-mem.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-router-splitk.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-split-cache-evict.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-promotion-followups.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-promotion-followups-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-meta-timing.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-moe-cache.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-moe-copy.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-moe-ep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-moe-range.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-native-1333.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-native-flash.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-p27b-3.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-p27b.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-peak.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-pmin-screen.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-prbe115-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-prefill-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-prefill-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-prefill-providers.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-profile-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-promote.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-q81-trace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-qfp21-dflash.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-qfp21.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-rccl-algo-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-rccl-screen.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-release-smoke.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-rerun-0910.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-routing.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-sched-split.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-smoke-v5.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-sparse-fa.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-sparse-proof.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-submit-perf.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-submit-split.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-synctrace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-threeway.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-threshold-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-topk-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-trim-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ts-skew-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ts23-ctx-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ub-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ubchunk.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1295.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1304.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1305.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1306.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1307.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1308.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1309.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1310.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-1311.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-balance.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-dg.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-dg2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-fusion-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-fusion-ab3.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-q81b.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v2-q81c.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v3-1312.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v3-1312c.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v3-1313.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v3-1314.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v4-abba.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v5-abba.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v5b-abba.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v6-ub.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-v7.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-vecq-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/quick-ab-depth.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/quick-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/rank-census.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/rccl-algo-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/review-with-model.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/routing-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/sched-split-summary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/smoke-models.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/spec-timing-summary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/submit-timing-summary.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/submit-timing-table.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/sync-tracer.c` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/threshold-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/topk-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/trim-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ub-1330.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ub-chunk.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ub-peak.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ub-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ubchunk-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/v7-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/vecq-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/wait-gpus-free.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/hi24-slice-a/verify_slice_a.py` | **TRANSITIONAL** | RA13 plan-owned lab implementation retained behind the documented compatibility wrapper until that entry point is retired. |
| `tools/lab/hi34-residency-gates/residency_gates.py` | **TRANSITIONAL** | RA12 plan-owned lab implementation retained behind the documented compatibility wrapper until that entry point is retired. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch-mtp.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch64k.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/patch1000/patch1000_verification.py` | **TRANSITIONAL** | PA43: patch 1000 backend-ops/llama-bench verification helpers moved verbatim out of `bigcherry.patch.validation_campaign` (never dispatched by the production CLI); loaded by path from `run_pa35_step1.py` and `tools/tests/patch/test_patch1000_verification.py`. |
| `tools/lab/patch1000/run_pa35_step1.py` | **TRANSITIONAL** | PA35 step 1: one-off gfx1201 hardware-evidence driver for patch 1000 (control vs subject backend-ops Q2_K/Q6_K correctness + perf); per GPT `req_71c1aaa166f446a1`, deliberately not shared production code. |
| `tools/lab/plan-qualification/activation-check.sh` | **TRANSITIONAL** | Plan-qualification activation probe used only by the legacy qualification queue; retire with that queue at jobs cutover. |
| `tools/lab/plan-qualification/contention_check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/cooldown.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/linux-test-suite.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/locked-run.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/make-serial-2.sh` | **TRANSITIONAL** | Writes the second plan-qualification job batch from environment-provided host paths. |
| `tools/lab/plan-qualification/noise.py` | **TRANSITIONAL** | Per-round paired-lane view (outlier rounds, per-arm CV) for campaign noise triage. |
| `tools/lab/plan-qualification/pcie-link-check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/pcie-retrain-job.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/preflight-fire.sh` | **TRANSITIONAL** | Plan-qualification firing gate; owns host/GPU locking and delegates one traced activation probe. |
| `tools/lab/plan-qualification/profile_run.sh` | **TRANSITIONAL** | PVPS10 kernel-coverage profile wrapper (queue `PROFILE` job type) over `bigcherry.patch.campaign.profile`. |
| `tools/lab/plan-qualification/queue-linux-tests.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/queue.sh` | **TRANSITIONAL** | Sequential per-GPU-lane runner for plan-qualification campaign jobs (restartable; skips finished runs). |
| `tools/lab/plan-qualification/retry-all.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/retry-failed.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/run_campaign.sh` | **TRANSITIONAL** | RDNA/nasone plan implementation loop: one-GPU validation-campaign launcher with host paths from env and output under work/; graduate into a campaign CLI verb or archive when the loop ends. |
| `tools/lab/plan-qualification/summarize.py` | **TRANSITIONAL** | One-line-per-run summary of plan-qualification campaign results (checks, lane effects, contract verdicts). |
| `tools/lab/plan-qualification/thermal-log.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/topk_backend_sampling_check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/withdraw_discarded.py` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/work-root.sh` | **TRANSITIONAL** | Resolves the campaign work root (env var, else environment.local.toml [env], else work/) for the plan-qualification scripts. |
| `tools/lab/prbe20-rd26-bisect/bisect_ubatch.py` | **TRANSITIONAL** | `prbe20-rd26-bisect` lab topic file (see `tools/lab/prbe20-rd26-bisect/README.md`); experiment-only, disposed per that README. |
| `tools/lab/radiance/build.sh` | **TRANSITIONAL** | `radiance` lab topic file (see `tools/lab/radiance/README.md`); experiment-only, disposed per that README. |
| `tools/lab/radiance/fetch-gcc14.sh` | **TRANSITIONAL** | `radiance` lab topic file (see `tools/lab/radiance/README.md`); experiment-only, disposed per that README. |
| `tools/lab/radiance/run-radiance.sh` | **TRANSITIONAL** | `radiance` lab topic file (see `tools/lab/radiance/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/ar-latency.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/ar-latency.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/bidir-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/bidir-check.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/both-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/build-rccl-nohostcall.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/cpu-root-ar.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/cpu-root-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/h2d-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/large-host-ar.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/large-host-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/p2p-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-ar-latency.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-rccl-env-sweep.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-rccl-nohostcall.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/rccl-env-sweep.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/reference-vllm/bench-openai.py` | **TRANSITIONAL** | `reference-vllm` lab topic file (see `tools/lab/reference-vllm/README.md`); experiment-only, disposed per that README. |
| `tools/lab/reference-vllm/run-llamacpp-r9700.sh` | **TRANSITIONAL** | `reference-vllm` lab topic file (see `tools/lab/reference-vllm/README.md`); experiment-only, disposed per that README. |
| `tools/lab/reference-vllm/run-radiance.sh` | **TRANSITIONAL** | `reference-vllm` lab topic file (see `tools/lab/reference-vllm/README.md`); experiment-only, disposed per that README. |
| `tools/lab/results/harvest.sh` | **TRANSITIONAL** | `results` lab topic file (see `tools/lab/results/README.md`); experiment-only, disposed per that README. |
| `tools/lab/results/summarize.py` | **TRANSITIONAL** | `results` lab topic file (see `tools/lab/results/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/mock_pipeline.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/mock_series_binding.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/real_bigcherry_process_smoke.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/real_bundle_failure_smoke.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/real_recovery_smoke.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/service_recovery_smoke.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/slurm_noble_smoke.sh` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/slurm_noble_v1_smoke.sh` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/slurm_noble_v2_smoke.sh` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/slurm_noble_v3_smoke.sh` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/run-campaign-durability/tree_activity_race_smoke.py` | **TRANSITIONAL** | `run-campaign-durability` lab topic file (see `tools/lab/run-campaign-durability/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-build.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-rocm-shim.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-routing.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-smoke.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-swift-bench.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-swift.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/bcop37-system.sh` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/iq_pack-ud.diff` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/routing-balance.py` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/strata/routing-skew.py` | **TRANSITIONAL** | `strata` lab topic file (see `tools/lab/strata/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/probe-27b-vk.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/queue-vk-ar.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/queue-vk-first.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/residency_gates.py` | **MOVE** | HI34 plan-specific gate moved to non-package `tools/lab/hi34-residency-gates/`; root wrapper retained for tests/legacy CLI. |
| `tools/rocm-env.ps1` | **MOVE** | Environment bootstrap; canonical destination tools/env/ in TR05. |
| `tools/rocm-env.sh` | **MOVE** | Environment bootstrap; canonical destination tools/env/ in TR05. |
| `tools/verify_slice_a.py` | **MOVE** | HI24 plan-specific verifier moved to non-package `tools/lab/hi24-slice-a/`; root wrapper retained for tests/legacy CLI. |

## Exit status

TR00 inventory is captured and both pre-existing test/check issues from the original baseline are now dispositioned: one resolved upstream (RD19/PA05), one fixed directly (subject-digest test fixture). No evidence was fabricated and no other actor's catalog decision was silently reversed.

## Change Log

- 2026-08-28T01:48:44.131940+00:00 (updated-by): Updated (no visible changes)

## Ledger-events

- chg_20260828_015352_published-the-active-docs-refe_7868
- 2026-08-28T01:53:52.318064+00:00 (updated-by): Updated: section:ledger-events
- chg_20260828_015434_aligned-active-planning-record_5360
- 2026-08-28T01:54:34.237199+00:00 (updated-by): Updated: section:ledger-events
