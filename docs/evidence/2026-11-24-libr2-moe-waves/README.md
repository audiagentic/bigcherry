# RDNA2 expert-GEMM wave-budget qualification

Status: **NOT READY**. This note records measurements and acceptance gates, not a shipment decision.

## Resource gate (first candidate)

The shipped gfx1030 plugin contains 404 kernels, including 69 grouped expert-GEMM instantiations. All 69 expert GEMMs spill, with 48–864 private bytes per thread. The first candidate scales the existing wave hint by the architecture's 1024/1536 VGPR-file ratio using a constexpr rule. All 69 expert GEMM resource records changed and all 335 other kernel records stayed unchanged; 53 expert GEMMs still spill, so this candidate does not pass the resource gate. Alternatives must be measured before any shipping recommendation.

## Host build and tests

Full libr2 plugin build: PASS, all three units, linked final module.

The following remote-MoE tests were built, executed, and passed (7/7):
`r3_moe_shm_test`, `r3_moe_backend_test`, `r3_policy_config_test`, `r3_moe_epoch_test`, `r3_moe_fp8_ref_test`, `r3_rad46_rows_policy_test`, and `r3_moe_mailbox_lifetime_test`.

Host-only tests do not establish GPU bit identity. The device bit-identity and cold replay gates are not yet run. No serving, ms/step, A/B, or KLD was run.
