---
id: PWR15
order: 0
plan: run-power-efficiency
state: in_progress
created-at: '2026-11-26T05:48:37.575642+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P0
work: M
---

# Harden existing platform power transaction and cross-owner recovery admission

## Description

Resolve the PWR13 P0 platform lease and exact restoration defects in existing PowerTransaction, without new writer or changing production promotion.

## Steps

1. Require lease before every write and completion, validate original controls before first write, verify all observed controls before terminal journal state. 2. Fail closed on malformed or reused journals and recovery identity. 3. Review native and platform journal ownership with RSC02/PWR08; enforce reciprocal admission before unattended candidate execution. 4. Run fault-injected fixtures then owner-approved hardware validation.

Current scope: recover XTX through native owner under lease; prove reciprocal native/platform admission and exact recovery, including physical restore and fault injection. Coordinate RSC02/PWR08.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files

tools/bigcherry/hardware/power_transaction.py, tools/tests/hardware/test_power_transaction.py; existing native owner only by coordination.

## Validation

Inject lease loss, unexpected control drift, malformed recovery inventory and stale state; no writes after lease loss; never resolve uncertain journal.

## Effort & Risk



## Standards



## Acceptance Criteria

No unattended candidate sweep until reciprocal owner admission, exact restoration and hardware qualification are proven. Keep unresolved journals on any uncertain write or restore; preserve existing supported controls and no-write observation.

Absorb PWR07/PWR04: explicit unsupported requests refuse before writes; optional absent/unreadable features warn, ordinary start records BDF/source/CPU and distinguishes missing sensor, permission and observe-only. No new writer, implicit apply or false cap=0 success. No candidate writes until safety gates pass.

## Notes

2026-11-26: hardened platform PowerTransaction: validate run/BDF and requested setting, reject reused/unknown/malformed journals, check fresh original snapshot and restorable range before writes, lease check per write and final apply/restore, full control plus physical identity/default cap readback for all touched devices, late caller readback invalidates terminal journal, strict recovery inventory. Historical active journals without saved identity require manual owner recovery; resolved historical records retained. Card-free hardware and CLI tests 289 passed/1 skipped. Native C++ already has per-write lease checks; cross-owner directories remain separate. PWR15 remains in progress: reciprocal admission across native and platform roots and approved real fault/recovery qualification still outstanding. No physical power writes or automated candidates in this slice.

2026-11-27: native 210 W diagnostic showed second XTX restore write fault, despite complete independent snapshots matching. Retain unresolved journal and recover through native owner. See PWR16 and docs/evidence/2026-11-26-power-session-closeout. Cross-owner admission remains open.

2026-11-27: consolidate portable power safety and startup observability from PWR07 into PWR15; retain existing acceptance and historical evidence.

2026-11-27 native owner hardening: full represented-control checks and final fleet readback on both shader control kinds; legacy absolute journals recover identity from bound original receipt; receipt mismatch refuses before writes; recovery disqualifies entire transaction; sysfs command errors identify command and errno, short writes fail closed. Native card-free acceptance executable passes all five groups including four golden profiles, recovery and fault injection, with contract fingerprint matched. Operational recovery and reciprocal journal admission remain open; no hardware writes in this slice.

2026-11-27: native D1 recovery PASS under exclusive GPU lease using new bc-serve power-recover. Pre-recovery journal and receipt byte-identical to original archived copies. Independent full physical identity and controls for all four GPUs matched original before and after. Native exit 0, journal and receipt restored, original failure and recovery events retained, non-qualifying. Lease free. Raw proof retained in work-root archive PWR15-native-recovery-20261127; managed harvest blocked by remote policy. Native component build and acceptance pass; 140 targeted pytest cases pass. Shared-root reciprocal admission implemented, offline tested; legacy private roots inventory and reviewed release deployment remain gates. No new candidate settings or production promotion.

2026-11-27 PR #1701 initial full CI found three host environment contract failures and two simulated hardware fixtures missing identity. Declare BC_WORK_ROOT as owner infrastructure (not a manifest setting); add stable fake identities without relaxing real admission. Targeted CI reproduction: 165 Python tests and 61 subtests passed; native transaction, manifest power and session group 1 passed. GPU held by another model run; no candidate hardware writes.

2026-11-27 follow-up: read-only historical journal inventory across both known shared work roots and retained live PWR scratch directories: 167 JSON journals across ten directories, all state restored; no unresolved state or parse errors. Default legacy XDG journal directory absent. This is a scoped inventory, not a proof of no arbitrary private roots.

2026-11-27 same-lease race addressed with existing host-local lock pattern at shared power/owner.lock: native owner holds across session, platform prepare/apply/restore/recover fence durable mutations; stale in-memory record and conflicting peer recovery refuse. Native recovery checks shared plus original root. Native component rebuild and three CTest targets pass, including contention; hardware/CLI and core guard suite 328 passed, one skipped, 61 subtests. PR #1701 merged; follow-up lock not deployed. Physical acceptance cannot run while unrelated model owns GPU lease.

2026-11-27 PRs 1701 and 1704 merged with full CI success. Physical platform 210W efficient memory and burst brackets passed through merged reciprocal admission/lock, exact all-device originals. Published bc-radiance-2020.23.1 tag source 83196d12 predates both owner fixes; do not equate it with updated native owner. Release deployment and physical native admission/restore remain open.

## Change Log

- 2026-11-26T05:48:37.575642+00:00 (created-by): Created by agent
- 2026-11-26T05:48:42.356489+00:00 (state-transition): State: pending → in_progress
- 2026-11-26T05:57:33.132221+00:00 (updated-by): Updated: section:notes
- 2026-11-26T21:28:12.962663+00:00 (updated-by): Updated: section:notes

## Reviews

- RV4680
- 2026-11-26T23:12:00.089479+00:00 (updated-by): Updated: section:notes
- 2026-11-26T23:13:35.421669+00:00 (updated-by): Updated: section:steps, section:acceptance_criteria
- RV4693
- 2026-11-27T00:05:12.419653+00:00 (updated-by): Updated: section:notes
- 2026-11-27T00:15:12.379866+00:00 (updated-by): Updated: section:notes
- 2026-11-27T00:33:31.410965+00:00 (updated-by): Updated: section:notes
- 2026-11-27T00:35:58.278415+00:00 (updated-by): Updated: section:notes
- RV4694
- 2026-11-27T00:41:22.690717+00:00 (updated-by): Updated: section:notes
- 2026-11-27T02:53:57.206402+00:00 (updated-by): Updated: section:notes
