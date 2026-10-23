---
id: RAD18
order: 0
plan: patching-radiance-rdna3
state: pending
created-at: '2026-10-21T23:17:12.131804+00:00'
breadth: ''
skill: ''
created-by: agent
---

# hc_up_mix_wide_kernel PF2 RT sweep (occupancy/register pressure)

## Description

Per GPT dev-agent review, sweep hc_up_mix_wide_kernel's RT (rows per workgroup, selects MW waves-per-block) at the PF2 call site to find whether occupancy/register pressure (not grid fill, which is already ample at this shape: n/kWideCH=160 column tiles) leaves real headroom in the existing fused BF16 up-mix kernel. Follows the RAD17 strategic pivot (GEMM-only and GEMM+separate-epilogue diagnostics both showed the naive hipBLASLt-replacement approach loses to the fused kernel).

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Context: this is a sibling investigation to RAD17 (FP8 down-product), targeting the BF16 up-mix kernel instead. See RAD17's notes for the full GEMM/epilogue diagnostic history that led here. Patch: rad0008_hc_upmix_rt_sweep, default-off via RAD_HC_UPMIX_RT env var (unset = bit-identical to current production RT=kPf2UpRT=256).

2026-10-22. rad0008 (PF2 up-mix RT=128 default, RAD_HC_UPMIX_RT=256 restores the original): kernel level vs unpatched build -2.1%/-3.2% of hc_read on the R9700 (M=2048/4096), -5.8%/-2.8% on the XTX, byte-identical. END TO END in the working layout (R9700 engine + 2 XTX helpers, ab-benchmark tools/lab/radiance/helper-upmix-rt-ab.json, one binary built with --experiment helper-layout-upmix-rt, 4 rotated pairs, --host-pool-mib 22528 in both arms because the 30 GiB pool plus the pinned n-gram table no longer fits free host RAM): pp4096 3351 (RT256) vs 3355 (RT128) tok/s (+0.1%), pp1024 3406 vs 3416 excluding one low RT128 run (+0.3%), tg2048 152.3 vs 152.2 -- no measurable end-to-end gain. Matches the share: hc_read is 8.9% of R9700 prefill device time in this layout (--profile-ops, 7306-token prompt, TTFT 3.49 s), so 2-3% off hc_read is ~0.2-0.3% end to end. Prefill device time in this layout: moe_mailbox_add 51.7% (helper wait + merge), gemm_nt_q 11.1%, hc_read 8.9%, moe_mailbox_publish 6.2%, local moe_gemm_q 5.9%. Next: R3_MOE_WORKER_TRACE decomposition of the mailbox wait (GPT req_bigcherry_ceee9968e4b5): exposed helper wait vs helper GPU time vs merge (the merge reads each helper's full-width bf16 result straight from mapped host memory). Hardware layout is fixed (owner).

## Change Log

- 2026-10-21T23:17:12.131804+00:00 (created-by): Created by agent
- 2026-10-22T07:32:45.588047+00:00 (updated-by): Updated: section:notes
