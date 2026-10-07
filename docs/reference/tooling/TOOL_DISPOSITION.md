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
`tools/bigcherry/check.py`; it is not a raw filesystem snapshot. A row can
intentionally name an ignored, historical, transitional, or machine-local path
when that path's ownership decision still needs to be carried forward. Do not
infer that every listed path is present or executable in the current checkout.

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
- `TRANSITIONAL`, `ARCHIVE`, and other non-`KEEP` dispositions describe an
  ownership decision, not an assertion that the path exists in every checkout.
- A move, retirement, graduation, or new lab file requires updating this
  registry, the owning plan/evidence references, and the focused hygiene or
  boundary checks in the same change.
- No implementation was moved or deleted merely to produce the TR00 baseline.
  Historical TR00 rows remain frozen under `docs/evidence/`; current changes
  must update this table without rewriting that evidence.

| Path | Disposition | Intended owner / rationale |
| --- | --- | --- |
| `.audiagentic/runtime/patch-system/psi_v2_block.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/download-msgpackr-prebuilds.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/download-msgpackr-prebuilds.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages-optional.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages-optional.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages-test.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages-test.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-gyp-build-optional-packages.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-which.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/node-which.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/uuid.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/uuid.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/yaml.cmd` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `.opencode/node_modules/.bin/yaml.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `artifacts/2026-08-21-hi35-hi36-27b-r9700/raw/pipeline.sh` | **TRANSITIONAL** | Evidence/reference-local harness; retain until caller and ownership audit completes. |
| `patches/_template/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0100_cmake_options/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0200_dispatch_hook/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0300_mmq_forced_j/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0400_mmvf_forced_block/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0500_mmf_forced_nwarps/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0600_mmvq_geometry/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0650_mmvq_native_variant/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0700_coverage_counters/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0800_server_shutdown_endpoint/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0810_replay_hit_diagnostics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0820_measurement_signature_shapes/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0830_split_reduce_telemetry/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/0900_pool_workspace_metrics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1000_rdna4_mmq_q2k_q6k_fix/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1002_hip_unsafe_math_opt_in/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1003_quantized_cpy_thread_block_fix/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1004_rms_norm_mul_rope_fusion/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1005_prompt_cache_checkpoint_selection/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1100_hi70_direct_op_evidence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1200_rd19_single_gpu_meta_bypass/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1201_rd20_attn_gate_tp_split/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1202_rd04_bf16_flash_attn_tile/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1207_rd17_moe_topk_down_fold/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1208_rd21_gfx1151_mmvq_nwarps_table/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1209_rd22_integrated_gpu_host_buffer_backout/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1210_rd26_bitidentical_decode_verify_standalone/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1215_rd394041_amd_stream_moe_overlap/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1216_rd43_concurrent_join_fusion_guard/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1217_rd44_graph_opt_default_rdna35/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1221_rd50_gdn_chunked_recurrence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1223_hi67_machine_readable_correctness_metrics/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1224_hi18_reduce_correctness_probe/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1225_hi85_nccl_heterogeneous_arch_guard/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1230_hip_autotune_inspect/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1231_hi14_graph_capture_lifecycle_evidence/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1232_hi81_windows_cxx_hipcc_flags_reach_compile/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1233_rd73_stable_graph_cache_key/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1234_rd58_pin_state_buffer_multigpu_restore/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1235_rd09_q81_activation_cache_foundation/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1236_hi105_deterministic_mul_mat_id_ids/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1237_rd30_moe_mmq_compact_grid/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1204_rd08_q6k_mmvq_vdr2/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1205_rd12_paired_mmvq_dual_output/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `patches/1206_rd13_mul_mat_add_view_fusion/patch.py` | **PACKAGE-LOCAL** | Patch-owned implementation or validation; maintain package-only production layout and refine in TR06. |
| `tmp/b1-breakdown.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-check.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-gate2-strict.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-gates.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-reject.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-spotcheck.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/b1-sweep.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/brutus-probe.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/dedupe-hi65.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/fix_test.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h35-s1b-equivalence.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-ab-stability.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-addon.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-brutus-pipeline.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-key-derivation.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-regret-analysis.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h36-whatif-misslog.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h65-runarm-test.bat` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h65-runarm.bat` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h65-telemetry.bat` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h65-typeperf-test.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h68-smoke-v2-analyze.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h68-smoke-v2.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h68-smoke-v3-analyze.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h68-smoke-v3.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/h68-smoke.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-analyze.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-analyze2.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-controls.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-gates-run.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-matrix-local.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-matrix.sh` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-noisefloor.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-select-controls.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/hi65-verdicts.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/pi-lens-shadow-probe.py` | **DELETE** | Plan-specific or scratch probe; deletion requires caller/reference proof in TR05. |
| `tmp/rd-bench-lane.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/rd04-pair-bench.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/rd04-pp-bench.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/re25_merge/A.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/re25_merge/base_fmt.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/re25_merge/base.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/re25_merge/merged.py` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/sliceA-configure.bat` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/t1.bat` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/tune-off-targeted.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/tune-off.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/tune-on.ps1` | **TRANSITIONAL** | Repository script outside canonical tooling tree; retain pending ownership audit. |
| `tmp/verify_slice_a.py` | **DELETE** | Plan-specific or scratch probe; deletion requires caller/reference proof in TR05. |
| `tools/lab/hi24-slice-a/verify_slice_a.py` | **TRANSITIONAL** | RA13 plan-owned lab implementation retained behind the documented compatibility wrapper until that entry point is retired. |
| `tools/lab/hi34-residency-gates/residency_gates.py` | **TRANSITIONAL** | RA12 plan-owned lab implementation retained behind the documented compatibility wrapper until that entry point is retired. |
| `tools/lab/plan-qualification/summarize.py` | **TRANSITIONAL** | One-line-per-run summary of plan-qualification campaign results (checks, lane effects, contract verdicts). |
| `tools/lab/plan-qualification/noise.py` | **TRANSITIONAL** | Per-round paired-lane view (outlier rounds, per-arm CV) for campaign noise triage. |
| `tools/lab/plan-qualification/profile_run.sh` | **TRANSITIONAL** | PVPS10 kernel-coverage profile wrapper (queue `PROFILE` job type) over `bigcherry.patch.campaign.profile`. |
| `tools/lab/plan-qualification/make-serial-2.sh` | **TRANSITIONAL** | Writes the second plan-qualification job batch from environment-provided host paths. |
| `tools/lab/plan-qualification/queue.sh` | **TRANSITIONAL** | Sequential per-GPU-lane runner for plan-qualification campaign jobs (restartable; skips finished runs). |
| `tools/lab/plan-qualification/run_campaign.sh` | **TRANSITIONAL** | RDNA/nasone plan implementation loop: one-GPU validation-campaign launcher with host paths from env and output under work/; graduate into a campaign CLI verb or archive when the loop ends. |
| `tools/lab/plan-qualification/work-root.sh` | **TRANSITIONAL** | Resolves the campaign work root (env var, else environment.local.toml [env], else work/) for the plan-qualification scripts. |
| `tools/bigcherry/__init__.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/__main__.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/ab_benchmark.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/analyze_gaps.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/artifacts.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/autotune_catalog.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/autotune_schema.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/bandit_simulator.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/builds.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_build.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_execution.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_graph.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_lane.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_plan.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_planner.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_resolution.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_source.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign_workers.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/campaign.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/candidate_binary_size.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/check.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/compare_tunes.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/comparisons.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/compile_check.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/config.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/context.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/correctness_evidence.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/csource.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/device_state_validate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/doctor.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/e2e_smoke_campaign.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/e2e_smoke_report.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/experiment_bundle.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/experiment_contract.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/focal_source_plans.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/generalise.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/generated_tree.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/graph_lifecycle_evidence.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/hi16_forced_native_parity.py` | **GRADUATE** | Generic native-versus-forced parity mechanics; extract behind a maintained correctness API without retaining HI16 naming. |
| `tools/bigcherry/hi18_run_corpus.py` | **TRANSITIONAL** | HI18-specific corpus runner; separate reusable reduction correctness from plan-specific corpus data before graduation. |
| `tools/bigcherry/hi80_generate_correctness_evidence.py` | **GRADUATE** | Generic correctness-evidence generation mechanics; extract behind the maintained evidence API before retiring historical entrypoint. |
| `tools/bigcherry/identity_separation.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/impact.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/inventory.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/kernel_fraction.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/lifecycle.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/moe_hostile_routing_sweep.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/moe_routing_gen.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/multi_gpu_validate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/parity_loaders.py` | **TRANSITIONAL** | Historical parity/cutover helper; retain until permanent invariant ownership and zero-caller proof. |
| `tools/bigcherry/parity.py` | **TRANSITIONAL** | Historical parity/cutover helper; retain until permanent invariant ownership and zero-caller proof. |
| `tools/bigcherry/patch_activation.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_catalog.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_lifecycle.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_registry.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_source_isolation.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_validation_campaign.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_validation_evidence.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patch_validation.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patcher.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/patchset.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/paths.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pin_status.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pin_transition.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pipeline.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/pool_protocol.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/promotion_correctness_gate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/promotion.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/provenance.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/rank_replay.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/ranking_policy.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/rd08_correctness_evidence.py` | **PACKAGE-LOCAL** | Moved to `patches/1204_rd08_q6k_mmvq_vdr2/validation/rd08_correctness.py`; global implementation deleted after retaining semantic tests. Wiring remains deferred until source/build pairing is guaranteed. |
| `tools/bigcherry/re14_real_run.py` | **TRANSITIONAL** | Historical acceptance harness; preserve campaign/artifact invariants before retirement. |
| `tools/bigcherry/re15_acceptance_run.py` | **TRANSITIONAL** | Historical acceptance harness; preserve campaign/artifact invariants before retirement. |
| `tools/bigcherry/re15_tamper_evidence.py` | **TRANSITIONAL** | Historical tamper/evidence harness; retain until integrity checks have a permanent owner. |
| `tools/bigcherry/recipes.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/reduce_correctness.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/release_validate.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/releases.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/replay_build_audit.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/replay_cache.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/replay_inspect.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/report.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/resource_report.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/resources.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/rocprof.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/runtime_smoke.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/signature_correctness_mapping.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/source_audit.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/source_identity.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/sources.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/symbol_map.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/telemetry.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/toolchain.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/transform_loader.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/transform_records.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/tune_journal.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/tune_promotion.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/upstream.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/validate_rd_patches.py` | **TRANSITIONAL** | Historical RD validator; generic validation remains authoritative. |
| `tools/bigcherry/vk_autotune_types.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/workspace.py` | **MOVE** | Maintained product or shared foundation; move mechanically during TR03/TR04/TR09/TR10. |
| `tools/bigcherry/analysis/candidate_report.py` | **KEEP** | Maintained analysis implementation; invoke with `PYTHONPATH=tools python -m bigcherry.analysis.candidate_report`. |
| `tools/pi-lens-shadow-probe.py` | **DELETE** | Unreferenced scratch probe; zero current callers/references confirmed during TR05.
| `tools/residency_gates.py` | **MOVE** | HI34 plan-specific gate moved to non-package `tools/lab/hi34-residency-gates/`; root wrapper retained for tests/legacy CLI. |
| `tools/rocm-env.ps1` | **MOVE** | Environment bootstrap; canonical destination tools/env/ in TR05. |
| `tools/rocm-env.sh` | **MOVE** | Environment bootstrap; canonical destination tools/env/ in TR05. |
| `tools/tests/test_ab_benchmark.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_analyze_gaps.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_ancestry_check.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_artifacts_provenance.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_autotune_catalog_compile_input_stability.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_bandit_simulator.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_blake2b_cross_lang_vectors.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_blas_plan_contract.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_build_descriptor.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_build_flip.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_build_identity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_build_flip.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_build.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_cutover_audit.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_execution.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_graph.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_lane.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_plan.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_planner.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_resolution.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_source.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_campaign_workers_build.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_candidate_binary_size.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_candidate_coverage.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_catalog_resource_blacklist.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_catalog_snapshot.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_check.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_cli_audit_stage.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_cli_patches_catalog_filter.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_cli_portability.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_compare_tunes.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_comparisons_promotion.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_config_v2.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_config.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_context.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_correctness_evidence.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_coverage_patch_anchor.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_db_migration.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_device_state_validate.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_dispatch_safety.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_e2e_smoke_campaign_identity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_e2e_smoke_campaign_s3b.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_e2e_smoke_report_bench_validation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_experiment_bundle.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_experiment_contract_cli.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_experiment_contract.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_external_sources.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_focal_source_plans.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_generalise.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_generated_layout.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_generated_tree.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_graph_lifecycle_evidence.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi09_catalog_completeness.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi101_workload_cache.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi16_correctness_reference.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi16_forced_native_parity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi17_blas_runtime_seam.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi18_reduce_correctness_probe.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi18_run_corpus.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi24_canary_summary.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi24_double_native.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi24_hot_list_py.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi24_hot_list.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi25_readiness.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi26_offline_readiness.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi34_flush.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi37_workload_digest.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi37_workload_overlap.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi53_native_wrapper_parity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi65_pre_sample.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi67_correctness_evidence_schema.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi67_correctness_metrics_patch.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi67_deterministic_seed_patch.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi68_canary_decision.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi68_probe_contract.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi69_correctness_timing.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi73_reachability.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi80_generate_correctness_evidence.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi85_nccl_heterogeneous_arch_guard.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi92_dispatch_counters.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi93_hardware_identity_cache.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi97_runtime_flat.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_hi99_tuner_config_macro.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_identity_separation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_impact.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_inventory.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_kernel_fraction.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_moe_hostile_routing_sweep.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_moe_routing_gen.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_multi_gpu_validate.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_native_select_timing.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_overlay_sync_audit.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_parity_loaders.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_parity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_1233_rd73_graph_cache_key.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_1234_rd58_pin_state_buffer.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_1236_hi105_deterministic_mul_mat_id_ids.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_activation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_catalog.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_explain_graph.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_governance.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_lifecycle.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_migrations.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_registry.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_resolution.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_selection.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_source_isolation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_validation_campaign_trace_probes.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_validation_contract.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_validation_evidence.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_validation_plan_integration.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patch_validation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_patcher.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_pin_status.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_pipeline.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_pool_protocol.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_promotion_correctness_gate.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rank_replay.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_ranking_policy.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rd08_correctness_evidence.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rd09_q81_cache_foundation.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rd30_hostile_routing.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rd54_mmvq_narrow_moe_coverage.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re04_materialization_safety.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re07_build_identity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re07_smoke_bundle_consumption.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re08_provenance_import_boundary.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re09_schema_v4.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re10_lifecycle.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re12_comparisons.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re13_promotion_wiring.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re25_3_sticky_taint.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_re25_artifact_descriptors.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_recipes.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_reduce_correctness.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_release_validate.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_releases.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_replay_cache_promotion_gate.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_replay_cache_wire.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_replay_inspect.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_replay_v5.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_report.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_residency_gates.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_resource_report.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_rocprof.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_runtime_smoke.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_signature_correctness_mapping.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_source_audit.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_source_identity.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_source_plan_patch_contract_links.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_split_reduce_telemetry.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_telemetry.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_transform_loader.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_transform_records.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_tune_journal.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_tune_promotion.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_tuner_artifact_json.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_upstream.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_variant_params.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_verify_slice_a.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_vk_autotune_types.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_vulkan_audit.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/tests/test_workspace.py` | **KEEP** | Permanent test coverage; domain reorganisation deferred to TR11. |
| `tools/verify_slice_a.py` | **MOVE** | HI24 plan-specific verifier moved to non-package `tools/lab/hi24-slice-a/`; root wrapper retained for tests/legacy CLI. |

## Planning capability rebaseline v3 temporary tooling

| Path | Disposition | Owner and rationale |
| --- | --- | --- |

## HI172 kernel-trace investigation tooling

| Path | Disposition | Owner and rationale |
| --- | --- | --- |
| `tools/lab/hi172-kernel-trace/config-27b-dual-xtx.json` | **TRANSITIONAL** | THA16/HI172: retained diagnostic campaign configuration for the completed dispatch-overhead investigation. |
| `tools/lab/hi172-kernel-trace/config-gpu0-longgen.json` | **TRANSITIONAL** | THA16/HI172: retained diagnostic campaign configuration for the completed dispatch-overhead investigation. |
| `tools/lab/hi172-kernel-trace/config-gpu0-realprompt.json` | **TRANSITIONAL** | THA16/HI172: retained diagnostic campaign configuration for the completed dispatch-overhead investigation. |
| `tools/lab/hi172-kernel-trace/config-gpu0.json` | **TRANSITIONAL** | THA16/HI172: retained diagnostic campaign configuration for the completed dispatch-overhead investigation. |
| `tools/lab/hi172-kernel-trace/run.py` | **TRANSITIONAL** | THA16/HI172: reusable diagnostic runner retained as investigation provenance; not a production benchmark engine. |
| `tools/lab/hi172-kernel-trace/run_perf.py` | **TRANSITIONAL** | THA16/HI172: reusable perf diagnostic runner retained as investigation provenance; not a production benchmark engine. |

## RD87 hipBLASLt oracle tooling

| Path | Disposition | Owner and rationale |
| --- | --- | --- |

## HI168 retained investigation tools

| Path | Disposition | Owner and rationale |
| --- | --- | --- |
| `tools/lab/patch1000/run_pa35_step1.py` | **TRANSITIONAL** | PA35 step 1: one-off gfx1201 hardware-evidence driver for patch 1000 (control vs subject backend-ops Q2_K/Q6_K correctness + perf); per GPT `req_71c1aaa166f446a1`, deliberately not shared production code. |
| `tools/lab/patch1000/patch1000_verification.py` | **TRANSITIONAL** | PA43: patch 1000 backend-ops/llama-bench verification helpers moved verbatim out of `bigcherry.patch.validation_campaign` (never dispatched by the production CLI); loaded by path from `run_pa35_step1.py` and `tools/tests/patch/test_patch1000_verification.py`. |

Inventory count: 403 script/tool files (vendor, build/cache, and artifacts excluded).

## Baseline blockers: reviewed and dispositioned (2026-08-25)

- RD19 evidence-state defect: resolved independently under **PA05** (completed) — owner deliberately demoted `1200_rd19_single_gpu_meta_bypass` from `validated` back to `untested` rather than fabricate HI83 hardware evidence. `bigcherry check --quick` now passes (`patch-catalog: ok`). Tracking review **RV82** closed as incorporated.
- `overlay.vendor_sync` (default/full check): remains a live, unrelated finding — 7 `ggml-cuda/hip-autotune-*` files differ between `src/` and the compiled `vendor/llama.cpp` tree at review time, consistent with in-progress uncommitted edits elsewhere in this working tree. Not a TR00 defect; not repaired here.
- Legacy flat subject-digest test failure (`test_legacy_flat_without_state_uses_implementation_identity`): root-caused and fixed. It was a Windows-only test-fixture bug — the fixture wrote its file via `path.write_text(...)`, which Windows silently translates `\n` to `\r\n` on disk, desyncing the raw-byte comparison (`_sha256_file`) from `patch_validation_subject_digest`'s text-mode (universal-newline) read. Fixed by writing the fixture with `path.write_bytes(...)` instead; no change to `patch_validation_subject_digest` itself. Full module (25 tests) now passes.
- Windows symlink-privilege error (`test_rv80_symlink_escape_rejected`): confirmed pre-existing and environment-specific (also independently logged in RD30.md); left as-is, not a TR00 concern.

## RHA12 pin-bump validation

| Path | Disposition | Owner and rationale |
| --- | --- | --- |
| `tools/lab/bump-validation/run_bump_validation.py` | **KEEP** | RHA12: standing bump-validation matrix (PIN_BUMP.md step 5/6) -- builds fresh at the current pin and launches the real production runtime-profiles across every real GPU individually plus the real dual-XTX multi-GPU topology; run on every future bump, not a one-shot experiment. |
| `tools/lab/bump-validation/smoke_worker.py` | **KEEP** | RHA12: the real per-cell delegate_argv worker run_bump_validation.py's runtime-matrix cells launch -- reuses ServerRunner to actually start each server, wait for /health, send one real completion, and shut down cleanly; part of the same standing bump-validation tool, not a one-shot experiment. |
| `tools/lab/ar-accuracy/gates.py` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/kld.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-ar-accuracy.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-kld-adaptive.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-kld-decode.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/ar-accuracy/queue-p2p-accuracy.sh` | **TRANSITIONAL** | `ar-accuracy` lab topic file (see `tools/lab/ar-accuracy/README.md`); experiment-only, disposed per that README. |
| `tools/lab/deepseek/probe-v4-flash.sh` | **TRANSITIONAL** | `deepseek` lab topic file (see `tools/lab/deepseek/README.md`); experiment-only, disposed per that README. |
| `tools/lab/deepseek/queue-v4-flash.sh` | **TRANSITIONAL** | `deepseek` lab topic file (see `tools/lab/deepseek/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/probe-27b.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-dflash.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/layout-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-probe.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-routing.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/routing-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch-mtp.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/native-vs-patched/server-ab-adaptive-switch64k.json` | **TRANSITIONAL** | `native-vs-patched` lab topic file (see `tools/lab/native-vs-patched/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/contention_check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/cooldown.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/locked-run.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/pcie-link-check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/pcie-retrain-job.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/retry-all.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/retry-failed.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/thermal-log.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/topk_backend_sampling_check.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/withdraw_discarded.py` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/prbe20-rd26-bisect/bisect_ubatch.py` | **TRANSITIONAL** | `prbe20-rd26-bisect` lab topic file (see `tools/lab/prbe20-rd26-bisect/README.md`); experiment-only, disposed per that README. |
| `tools/lab/profiling/profile-27b-q8.sh` | **TRANSITIONAL** | `profiling` lab topic file (see `tools/lab/profiling/README.md`); experiment-only, disposed per that README. |
| `tools/lab/profiling/queue-27b-profile.sh` | **TRANSITIONAL** | `profiling` lab topic file (see `tools/lab/profiling/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/r9700-g1.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/r9700-g2.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/r9700-g3.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/r9700-g4.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx1-g1.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx1-g2.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx1-g3.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx1-g4.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx2-g1.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx2-g2.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx2-g3.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/configs/xtx2-g4.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/iq-mmvq-xtx1-iq3_xxs.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/iq-mmvq-xtx1-iq4_xs.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/kquant-f32-xtx1-q4_k_m.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/kquant-f32-xtx1-q6_k.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/kquant-f32-xtx2-q4_k_m.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/kquant-f32-xtx2-q6_k.json` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/make-configs.py` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/queue-iq-mmvq.sh` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/queue-kquant-f32.sh` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/quant-sweep/queue-quant-sweep.sh` | **TRANSITIONAL** | `quant-sweep` lab topic file (see `tools/lab/quant-sweep/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/build-rccl-nohostcall.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/h2d-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/p2p-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-rccl-env-sweep.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-rccl-nohostcall.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/rccl-env-sweep.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
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
| `tools/lab/dflash/queue-27b-detail.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-q4.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/linux-test-suite.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/plan-qualification/queue-linux-tests.sh` | **TRANSITIONAL** | `plan-qualification` lab topic file (see `tools/lab/plan-qualification/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/probe-27b-vk.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/queue-vk-first.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/lab/vulkan/queue-vk-ar.sh` | **TRANSITIONAL** | `vulkan` lab topic file (see `tools/lab/vulkan/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-ar.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ar-trace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-ar-trace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/ar-latency.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/ar-latency.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/queue-ar-latency.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/cpu-root-ar.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/cpu-root-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/both-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/large-host-ar.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/large-host-ar.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/bidir-check.hip` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/rccl/bidir-check.sh` | **TRANSITIONAL** | `rccl` lab topic file (see `tools/lab/rccl/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-flash-cpuroot.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chunk-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-chunk-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/threshold-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-threshold-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/long-ctx-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-profile.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/long-ctx-fit.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-fit.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/rccl-algo-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-rccl-algo-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-decode.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ctx-fit-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-long-ctx-perf.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/kpool-ab-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-ab-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/kpool-parity.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-parity.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/deploy-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-deploy-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/determinism.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-determinism.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/graphs-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-graphs-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-apitrace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/sync-tracer.c` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-synctrace.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/sync-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-sync-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/topk-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-topk-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-kpool-parity-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/balance-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-balance-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/fit-sweep-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fit-sweep-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-draft-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/ub-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ub-sweep.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-f16k.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-f16k.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search.py` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-search.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/trim-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-trim-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-ctx-fit-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/maxctx-search-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-maxctx-search-2.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/chain-serial.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-numeric.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/draft-quant.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-draft-quant.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/vecq-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-vecq-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/quick-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/combined-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-combined-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric-f16.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-numeric-2gpu.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-ref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-ref.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-v2-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/gather-v2-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-gather-v2-ab.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/flash-next/queue-fusion-quick.sh` | **TRANSITIONAL** | `flash-next` lab topic file (see `tools/lab/flash-next/README.md`); experiment-only, disposed per that README. |
| `tools/lab/dflash/queue-27b-cpuroot.sh` | **TRANSITIONAL** | `dflash` lab topic file (see `tools/lab/dflash/README.md`); experiment-only, disposed per that README. |

## Exit status

TR00 inventory is captured and both pre-existing test/check issues from the original baseline are now dispositioned: one resolved upstream (RD19/PA05), one fixed directly (subject-digest test fixture). No evidence was fabricated and no other actor's catalog decision was silently reversed.

## Change Log

- 2026-08-28T01:48:44.131940+00:00 (updated-by): Updated (no visible changes)

## Ledger-events

- chg_20260828_015352_published-the-active-docs-refe_7868
- 2026-08-28T01:53:52.318064+00:00 (updated-by): Updated: section:ledger-events
- chg_20260828_015434_aligned-active-planning-record_5360
- 2026-08-28T01:54:34.237199+00:00 (updated-by): Updated: section:ledger-events
