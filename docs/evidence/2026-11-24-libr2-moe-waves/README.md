# RDNA2 expert-GEMM wave-budget qualification

Status: **NOT READY** pending complete device bit-identity and cold-timing gates. Nothing is shipped.

## Implemented rule

The existing per-instantiation `moe_waves` hint is scaled from the larger register file to the RDNA2 register file, floored, and capped by the architectural number of full-width per-thread VGPR allocations: `min(max(1, floor(tuned * 1024 / 1536)), 1024 / 256)` on gfx1030 alone. The four numerical operands denote architectural register-file quantities, not model/shape-specific constants. All non-gfx1030 behavior is unchanged. There is no new instantiation table or change to GEMM arithmetic.

## Resource gate

The release and candidate gfx1030 code objects have the same 404 kernel names, 69 expert-GEMM instantiations and 335 unrelated instantiations. **Every unrelated resource record is identical.** The resource CSV covers all 404 names and records before/after VGPRs, SGPRs, SGPR/VGPR spills, per-thread private bytes, register-file VGPR wave capacity, and the expert-GEMM requested wave count. A separate 42-row CSV identifies every remaining expert-GEMM spill, with its exact resource counts. Attribute hints now demand three waves on 16 instantiations and four on 53.

| Candidate | Expert GEMMs with any SGPR/VGPR spill | Expert GEMMs with VGPR spills/private scratch | Sum of private bytes/thread across expert GEMMs |
| --- | ---: | ---: | ---: |
| Shipped baseline | 69/69 | 69/69 | 31,580 |
| Simple 1024/1536 scaled cap | 67/69 | 53/69 | 17,468 |
| No waves attribute (experiment only) | 58/69 | 22/69 | 5,108 |
| Register-file-derived final cap | **42/69** | **20/69** | **4,648** |

The remaining 20 VGPR-spilling instantiations all allocate the architectural maximum of 256 VGPRs/thread, with 148–364 private bytes/thread; their remaining live ranges cannot be accommodated by relaxing the wave cap. A further 22 instantiations have SGPR spills but no private scratch, while 27 have neither. The resource gate accepts an explicitly enumerated residual but does not establish performance or bit identity.

## Host build and tests

The full libr2 plugin configured, compiled all three HIP units, and linked successfully under low-priority single-job builds after the final resource rule. The remote-MoE host test configuration and targeted build succeeded. The following seven CTest cases actually ran and passed:

- `r3_moe_shm_test`
- `r3_moe_backend_test`
- `r3_policy_config_test`
- `r3_moe_epoch_test`
- `r3_moe_fp8_ref_test`
- `r3_rad46_rows_policy_test`
- `r3_moe_mailbox_lifetime_test`

These tests are host-only and do not establish GPU bit identity. The device identity and cold replay gates remain pending. No serving, ms/step, A/B, or KLD was run.
