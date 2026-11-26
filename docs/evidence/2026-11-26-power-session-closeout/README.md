# Power qualification session closeout, 2026-11-26

Audit on 2026-11-27. Scope: PWR12 through PWR16, independent
GPU screening and the native short model-run comparisons; linked
RCS27/RCS28 work. This is retention and handoff, not qualification.

## Evidence and source ownership

- [Inventory](inventory.json): 327 managed capture bundles, each verified
  with the existing harvest verifier, including all 238 run IDs cited
  by the power reports and plans. All 327 run manifests are harvested.
  Failed and exploratory runs are included. The index records bundle
  digests and original run states, not new pass verdicts.
- Raw evidence remains in the existing work-root evidence-hold store;
  the bundle ID is its locator. Do not replace or prune a referenced
  bundle because its worktree is gone.
- Local work archive PWR16-native-cap-20261126 retains exact available
  scratch runners/configurations, native run trees, probe builds and
  code objects, CI logs, supporting PWR/RCS scripts, and the verified
  Git bundle of all 30 prior branch tips. Inventory records 646 files
  and SHA256 digests. 441 scratch files and 80 work-root support files
  were copied and compared byte-for-byte. The originals were not deleted.
  Raw host-specific data and executables are not Git source.
- Production code, fixtures and tests are already merged through the
  earlier power and RCS pull requests, including PWR16 in #1685 and
  expanded physical evidence in #1691. Preserve original commit
  history in the Git bundle rather than replaying squash-merged branches.
  PWR13 precision draft additions were checked verbatim against main
  in both the code guide and AI_POWER_SCREEN. No duplicate merge needed.
- The earlier PWR10/RSC14 compact tracked acceptance report remains
  at [native acceptance](../2026-11-26-pwr10-rsc14-native-acceptance/README.md).
  Its original scratch directory was not present at this audit; do not
  claim a new raw reproduction of that earlier session.

## Latest capped inference

See [power-tuning evidence](../../reference/power-tuning-evidence-2026-11.md)
and [independent AI screen](../../reference/tooling/AI_POWER_SCREEN.md).
Capture 20261126T212407Z-native-power-cap-0cc14c765951 holds 112 exact
source artifacts plus summary (including native manifests, receipts,
journals, raw power samples, request hashes, configs, logs and runner
source), verified under capture/v1. Failed diagnostic, not admitted
production evidence. 16/16 inference requests matched by prompt/output
hash and token counts. Four independent BF16/FP32 comparison arms
pass 16/16 cases, 163232 GEMMs. Native comparison uses Radiance
2020.23.1 and the same Qwen27B TP3 model. Closing stock arm was not run.