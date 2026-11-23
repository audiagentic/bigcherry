# RDNA2 expert-GEMM wave-budget qualification

Status: **NOT READY**. Device bit identity passes but cold timing fails and independent complete host-reference coverage is outstanding. Nothing is shipped.

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

The remaining 20 VGPR-spilling instantiations all allocate the architectural maximum of 256 VGPRs/thread, with 148–364 private bytes/thread; their remaining live ranges cannot be accommodated by relaxing the wave cap. A further 22 instantiations have SGPR spills but no private scratch, while 27 have neither. The resource gate accepts an explicitly enumerated residual; the later sections separately establish bit identity and a failed performance gate.

## Host build and tests

The full libr2 plugin configured, compiled all three HIP units, and linked successfully under low-priority single-job builds after the final resource rule. The remote-MoE host test configuration and targeted build succeeded. The following seven CTest cases actually ran and passed:

- `r3_moe_shm_test`
- `r3_moe_backend_test`
- `r3_policy_config_test`
- `r3_moe_epoch_test`
- `r3_moe_fp8_ref_test`
- `r3_rad46_rows_policy_test`
- `r3_moe_mailbox_lifetime_test`

These tests are host-only; they do not themselves establish the GPU bit-identity result documented below. No serving, ms/step, A/B, or KLD was run.

## RDNA2 same-process bit-identity gate

The checked 2020.8.5 release and the candidate report ABI 44 with the same ABI-layout fingerprint `0x5cc71b9c5bb7ab40`. Both were loaded in one process, with only the gfx1030 device visible. The queued short measurement lease released after the run.

**PASS: 2,304 of 2,304 bit-identical complete BF16 output comparisons**. Row counts 1–256 were all exercised for gate/up (N1280, K2560, two parts, token order), down (N2560, K640, one part, sorted order) and a second gate/up geometry (N1536, K2048, two parts), with 512 experts and top-10 routes. Each shape used `w4a8h`, `w4nl64a8h`, and `w4nl32a8h` weights with E4M3 activations. 10,610,933,760 bytes compared. Shape output digests: `ed5bdb4eb6ca8943`, `33d97161529a5b83` and `22dfe7068b0df103`.

This proves the candidate exactly matches the shipped GPU result on those inputs; it **does not replace an independent host numeric reference**. Cold timing is reported below; full host-reference coverage is outstanding. No serving, A/B, KLD or ms/step run was performed.

## Cold replay and acceptance

**FAIL — NOT READY.** Three cache-flushed HIP-event measurements per arm were executed for all 54 combinations of the three validated geometries, three W4 forms, and M=1/4/8/16/32/64. Each sample had a 256 MiB intervening device write and synchronization, with alternating arm order. All timed output checks still matched bit-for-bit.

The median candidate is beyond the shipped repeat range in **26 of 54** combinations, with disjoint slower repeat ranges in 23. Just **6 of 54** are at least 10% faster. At 32 rows in the Flash-Next gate/up geometry, the native W4NL64 projection changes from 10,219 to 13,096 microseconds (0.780x throughput), and W4NL32 from 10,291 to 13,000 microseconds (0.792x). The standard W4 case improves at four rows from 2,033 to 1,705 microseconds (1.192x), but this is not consistent across formats or rows. `cold-replay.csv` records every per-case median, minimum, maximum and spread.

Register spills were reduced substantially, but the compulsory occupancy limit is likely too restrictive for some large-row or transformed-weight tiles. This attribution is a hypothesis. A future selectively guarded variant needs an architecture-derived per-instantiation resource and timing justification, followed by new build, bit, and replay checks. This variant is **not suitable for shipping**. The complete independent CPU host reference for all GPU-produced W4 outputs was not executed.
