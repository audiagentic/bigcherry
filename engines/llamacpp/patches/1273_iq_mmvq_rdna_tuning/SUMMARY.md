# 1273_iq_mmvq_rdna_tuning

**Status:** evaluated
**Plan item:** none

No hardware performance or correctness claim is made by this patch package.

## Scope

Single-token MMVQ (`ncols_dst == 1`) only, and only `IQ4_XS` / `IQ3_XXS` on the RDNA architecture tables used by gfx1100 and gfx1201. With both tuning env vars unset, b11233 launch geometry and vec-dot entry points are unchanged.

Independent A/B gates:

- `BIGCHERRY_IQ_MMVQ_VDR=1`: lower-VDR candidate.
- `BIGCHERRY_IQ_MMVQ_NWARPS=1`: per-architecture block-width candidate.
- `BIGCHERRY_PATCH_TRACE=1`: once-per-process activation evidence.

Candidate matrix:

| type | arch | pristine | VDR candidate | nwarps candidate |
|---|---|---:|---:|---:|
| IQ4_XS | gfx1100 | VDR4 / NW1 | VDR2 | NW2 |
| IQ4_XS | gfx1201 | VDR4 / NW8 | VDR2 | NW4 |
| IQ3_XXS | gfx1100 | VDR2 / NW1 | VDR1 | NW2 |
| IQ3_XXS | gfx1201 | VDR2 / NW1 | VDR1 | NW2 |

The lower-VDR vec-dot variants split the pristine work without changing quant math: IQ4_XS VDR2 selects the appropriate half of each q8 block; IQ3_XXS VDR1 selects the appropriate q8/sign half of the pristine VDR2 pair.

Activation marker:

`BIGCHERRY_PATCH_HIT patch=1273_iq_mmvq path=<vdr|nwarps|vdr_nwarps> type=<iq4_xs|iq3_xxs> arch=<gfx1100|gfx1201>`

The proposed IQ3_XXS `v_perm_b32` sign expansion is intentionally **not** included in this first patch: the launch experiments should establish whether decode is gather/cache-latency limited or VALU-unpack limited before adding an instruction-selection variable.

## rocprofv3 capture plan

Run the same decode workload and binary for each type+GPU with four arms: pristine (both envs unset), VDR only, nwarps only, VDR+nwarps. Warm up first; capture a fixed set of decode dispatches with a kernel include regex matching `mul_mat_vec_q` and keep model, prompt, generated tokens, clocks/power policy and visible device fixed.

Before capture, enumerate the actual counter spelling on each card with `rocprofv3-avail info --pmc` and validate each group with `rocprofv3-avail pmc-check`. Counter availability is architecture/ROCm-version dependent, so use the listed equivalent when a derived `_sum` name is absent.

Use separate passes rather than forcing incompatible counters into one pass:

1. **VALU/unpack pass**: `SQ_WAVES`, `SQ_INSTS_VALU`, `SQ_THREAD_CYCLES_VALU`, `SQ_ACTIVE_INST_VALU`, plus `SQ_INSTS_SALU` when available. Normalize instructions/cycles by `SQ_WAVES` and by decoded token.
2. **VMEM/table-gather pass**: `TCP_PERF_SEL_TOTAL_CACHE_ACCESSES`, `TCC_HIT`, `TCC_MISS`, `TCC_EA_RDREQ` (or corresponding aggregated/`_sum` counters), and `TCP_TCP_TA_DATA_STALL_CYCLES` when exposed. Also collect an available SQ VMEM/wait counter (`SQ_INSTS_VMEM`, `SQ_WAIT_INST_VMEM`, or nearest listed equivalent).
3. **Timing/occupancy pass**: kernel duration from kernel trace plus `SQ_WAVES`; retain profiler-emitted `VGPR_Count`, `SGPR_Count`, `Scratch_Size`, workgroup size and LDS size from counter output.

Interpretation:

- Table-gather limited: kernel time tracks TCC/TCP miss/read/stall or VMEM-wait changes while VALU instructions per wave remain roughly flat.
- VALU-unpack limited: kernel time tracks `SQ_INSTS_VALU` / `SQ_THREAD_CYCLES_VALU` changes while TCC/TCP traffic and miss ratio stay roughly flat.
- Occupancy/register limited: nwarps changes time materially while memory traffic is stable and emitted VGPR/scratch/workgroup data explains the occupancy shift.

Only after the above separates IQ3_XXS unpack cost from table-gather cost should a byte-permute sign-expansion arm be added; it must be a new independent env gate so launch tuning and sign expansion remain factorially separable.

## b11474 BPB01 review

Current-pin composition and disposition options are recorded in `releases/evidence/bpb01-four-evaluated.md`. This review does not change patch state.
