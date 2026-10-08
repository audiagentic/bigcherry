# BPB01: four evaluated patches at b11474

Pin: llama.cpp b11474 / b9acf138a1e2. This slice changes no patch state.

## 1250_nro01_allreduce_q8_wire

- **Source verified:** package requires 1252 P2P + 1272 compressed wire. No current `[experiment.*]` references 1250, so the requested `patch-rebase-check --source bigcherry --experiment <its experiment>` is not defined on main.
- **Last recorded result:** none in 1250 SUMMARY/README; the package records no BigCherry hardware performance result. Its required provider 1252 is rejected at b11474: P2P was re-tested on the lab topology on 2026-10-07, faults, and "does not compose at b11474 either"; its experiment/script were removed.
- **Overtaken:** validated 1291 cpu-root AllReduce is the production Flash-Next provider (+5-6% decode on its 0504396 adoption ABBA, greedy identical). 1272 independently owns compressed host wire; 1250 is only the P2P extension + optional residual fusion.
- **Proposed final disposition:** superseded/infeasible as a package, not rejected: its mandatory 1252 transport is rejected at the current pin and there is no current 1250 activation measurement. If PNRO02 residual fusion is still wanted, split it from the dead P2P dependency into a new package and measure it independently.

## 1273_iq_mmvq_rdna_tuning

- **Source verified:** experiment `iq-mmvq`; single-token (`ncols_dst==1`) IQ4_XS/IQ3_XXS only, gfx1100/gfx1201; independent VDR and nwarps gates; marker identifies arm/type/arch.
- **Last recorded result:** SUMMARY explicitly says no hardware performance or correctness claim. README says hardware was run, but contains no numbers/run id. Therefore there is no auditable measured result to quote.
- **Not overtaken:** 1345 optimizes MoE id-list construction at large prefill batches; 1350 targets ordinary few-tile Q8_0 MMQ. Neither owns 1273's single-token IQ MMVQ VDR/nwarps path. QFP38 explicitly leaves single-token IQ launch/VDR tuning in 1273.
- **Run now:** build production + experiment `iq-mmvq`, then three separated ABBAs on the same binary:
  ```bash
  BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_IQ_MMVQ_VDR=1" tools/lab/flash-next/queue-env-ab.sh bpb01-1273-vdr <build-run-id> 8192 24576 98304
  BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_IQ_MMVQ_NWARPS=1" tools/lab/flash-next/queue-env-ab.sh bpb01-1273-nw <build-run-id> 8192 24576 98304
  BIGCHERRY_PATCH_TRACE=1 AB_ENV="BIGCHERRY_IQ_MMVQ_VDR=1 BIGCHERRY_IQ_MMVQ_NWARPS=1" tools/lab/flash-next/queue-env-ab.sh bpb01-1273-both <build-run-id> 8192 24576 98304
  ```
  Require `BIGCHERRY_PATCH_HIT patch=1273_iq_mmvq` in B, identical greedy md5 A/B at all depths, and decode separation; prefill is a control. The lower-VDR path claims unchanged quant math, so a greedy mismatch fails the arm.

## 1275_ar_small_latency

- **Source verified:** current experiment `adaptive-wire-latency` contains 1272 + 1275. Marker: `patch=1275_ar_small path=small_ar`.
- **Last measured:** PGC10/PGC11, 2026-10-01, Qwen3.8-27B Q8_0 dual XTX with MTP: `slot_sync=none` neutral; `small_blocks=1` -8.5% decode. Both were dropped from the adaptive-wire matrix.
- **Overtaken for Flash-Next production:** 1291 cpu-root is the production AllReduce provider, so 1275's mapped-host internal-AllReduce latency controls are not the production path. The 27B default-AllReduce model remains the appropriate current-pin retest if a rejection decision is wanted.
- **Run now for a current-pin decision:** build `adaptive-wire-latency` for gfx1100 and use the same binary as A/B in `prod27b-ab.sh` with default AllReduce:
  ```bash
  tools/lab/flash-next/prod27b-ab.sh <llama-server> <same-llama-server> <out>/slot BIGCHERRY_PATCH_TRACE=1 BIGCHERRY_AR_SLOT_SYNC=none
  tools/lab/flash-next/prod27b-ab.sh <llama-server> <same-llama-server> <out>/b1 BIGCHERRY_PATCH_TRACE=1 BIGCHERRY_AR_SMALL_BLOCKS=1
  ```
  A = defaults, B = one changed gate. Require marker in B and greedy md5 identity. Rejection requires a current-b11474 regression/neutral result with the marker firing.

## 1293_sched_single_input_sync

- **Source verified:** current experiment `ar-cpu-root-kpool-sync`; b11474 rebase mechanics explicitly test that 1293 composes before production 1326.
- **Last measured:** 2026-10-03 `flashnext-sync-ab-1`, ABBA on 1291+1292: 10K ms/step base 50.4/50.8 vs new 50.6/50.1; 80K 65.0/65.4 vs 65.2/65.5 (neutral). `hipStreamSynchronize` calls per 256-token decode fell 40325 -> 32894 (-18%); no-MTP greedy output identical. Hardware review 2026-10-04 remained neutral.
- **Overtaken:** production 1326 async host inputs attacks the same scheduler submission/input-copy bottleneck at a stronger level: adoption ABBA +9.4% decode ~8K, +7.8% ~64K, submit timing 5.7 -> 3.0 ms/round, greedy identical.
- **Proposed final disposition:** superseded by 1326, not rejected. 1293 has no `BIGCHERRY_PATCH_HIT` marker, so it cannot satisfy the owner's current-pin rejection rule without changing the mechanism first. Its old neutral result plus the later validated 1326 gain justify retaining it only as historical cleanup evidence.

## Composition verification

PR CI is used as the current-pin source-composition check for the three configured experiments because changing these package summaries puts them in the offline audit scope. Exact `patch-rebase-check` results will be recorded here from that run. 1250 has no current experiment and is therefore reported as not runnable by the requested experiment command rather than inventing one.

## Verification boundary

Everything labelled source verified or measured above is quoted/paraphrased from current patch metadata, summaries/readmes, current recipes, or validated promotion records. Proposed dispositions and expected applicability are inference; no state changes are made.
