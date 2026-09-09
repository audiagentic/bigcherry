# PHA02 RCCL heterogeneous dispatch source audit

Date: 2026-09-09  
Host: Brutus (environment is referenced by host-local paths; no host
credentials or server configuration is committed)

## Scope and provenance

This is a source/runtime audit for PHA02 (successor of HI137). It does not
change BigCherry's RCCL guard or promote any RCCL patch. The RCCL source used
for the audit is the isolated Brutus checkout:

```text
/home/audumla/rccl-heterogeneous-src/rccl
revision: 57e58688f44c77076ad536ef1f6b68741fc6e694
```

The existing crash/qualification records remain historical evidence under
their original plan IDs. The following host-local artifacts were inspected;
their hashes make the external evidence identity explicit:

| Artifact | SHA-256 |
| --- | --- |
| `rq04-01/environment.txt` | `36d91dfad7870aba539b68e146026da137c752e4aa2172ce16e259698a2cffa8` |
| `rq04-01/rccl-target-controls.txt` | `8135fe200e25bdc3a5f785470efe60d23062b162ca31e1afe802044cd4d20f0e` |
| `rq04-01/rccl-code-objects.txt` | `a87eb26585998a0f679b548ad767ce376818a6e143e1198cb3bbe362a2380ec5` |
| `rq04-01/all-reduce-ldd.txt` | `52a3c70471dfb33c6f745ec97f1187d35d09d4af71b23b52fae0264e698c433e` |
| `rq08-01/cases.jsonl` | `f0f05df8a319b7cdd575854b499f286d105c068678c80cfee4cd2bdaf0641b1a` |

The source build's code-object listing contains `gfx1030`, `gfx1100`, and
`gfx1201`, so the observed failure is not explained by a missing target in the
inspected RCCL bundle. `all-reduce-ldd.txt` binds the qualification binary to
the isolated RCCL install rather than an unrelated system library.

## Source-level trace

The pinned source shows three relevant facts:

1. `src/init.cc:2150-2157,2220` records each rank's local
   `devProp.gcnArchName` in `comm->archName`.
2. `src/init.cc:1362-1364` derives `comm->topo->tuning` from that local
   architecture. `src/graph/tuning.cc:638-783` then uses that rank-local
   tuning index to populate the communication model and protocol/channel
   decisions.
3. `src/enqueue.cc:896-898` selects the generic kernel function only by
   unroll/trace index, and `src/enqueue.cc:1808-1832` launches that function
   with `hipExtLaunchKernel`. There is no BigCherry-visible per-rank code-object
   selector or safe error return at this launch boundary.

This is consistent with the historical HI85 trace: communicator topology and
per-rank tuning initialization can succeed, while the actual collective launch
fails later with a HIP-level invalid-device-function/illegal-state fault. A
return-code fallback at communicator initialization cannot recover from that
launch-time fault.

## Runtime controls and disposition

The source build was configured for a multi-architecture bundle and the
crash-isolated qualification harness preserved the homogeneous control and
post-fault recheck. The `rq08-01` matrix recorded, for each tested
XTX+R9700 topology, three correctness-passing cases, two explicitly
unsupported requests, and one timeout; these results are not promoted as a
universal heterogeneous RCCL guarantee. Existing HI85/HI138 evidence remains
the authority for the supported/unsafe topology boundary.

No safe RCCL source fix was identified in this audit. A real repair would need
to change RCCL's own per-rank device-kernel/code-object dispatch and then pass
the runbook's independent correctness, crash-freedom, and topology matrix
gates. BigCherry must not attempt to emulate that inside its tuning or patch
layers. The current production decision therefore remains: retain the
fail-closed admission guard, keep META as the safe heterogeneous path, and do
not promote an RCCL heterogeneous-dispatch patch from this audit alone.

## Matching-runtime source acquisition attempt

The historical failing runtime identified by the dev-GPT review is associated
with RCCL source revision
`9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe` (RCCL 2.27.7-1, build ID
`168a84…`). That object is not present in the Brutus checkout used above. A
fresh acquisition was attempted on 2026-09-09:

```text
cd /home/audumla/rccl-heterogeneous-src/rccl
git fetch --no-tags origin 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe
fatal: remote error: upload-pack: not our ref 9fb6fbe7bfb4df87c4c9a09b5bb5239670f04ffe
```

The installed multi-architecture library is available, but its source tree
cannot be mapped to that historical commit from the available remote. The
current source audit is therefore useful provenance and a bounded no-fix
finding, but it is **not** the exact-runtime source trace required to close
PHA02. PHA02 remains open pending either matching source/build acquisition or
a separately documented blocked/no-safe-change disposition.

## Reproduction commands

The authoritative commands and complete logs remain on Brutus under
`/home/audumla/bigcherry/artifacts/rccl-heterogeneous/rq04-01/` and
`rq08-01/`, driven by the checked-in
`tools/bigcherry/profiling/rccl_qualify_campaign.py` and governed by
`docs/reference/testing/RCCL_HETEROGENEOUS_RUNBOOK.md`.
