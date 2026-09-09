# PHA03 RCCL topology and runtime evidence snapshot

Date: 2026-09-10  
Execution environment: Brutus role (resolved through the configured
environment; no host connection details are committed)

This is an evidence snapshot for the fail-closed PHA03 plumbing. It does not
enable patch 1225 admission and it is not a positive topology allowlist.

## Runtime identity

The observed stack reported:

- ROCm/HIP: `7.2.53211-97f5574fe2`, install role `rocm-7.2.4`;
- RCCL: `2.27.7`, compiled with ROCm `7.2.4.0-93-97f5574fe2`;
- RCCL library SHA-256:
  `53149bc0a64faa580876b0f6b745ebfc02c4371be4e2a24daf90c23782ed96d3`;
- RCCL ELF build ID: `a1d6226519ab6358243c2fa0c315f8884f42da5b`;
- amdgpu module source version: `4F5C8C3B57F8DA15B66C039`.

## Observed placement graph

The four visible GPUs were observed as:

| Architecture | Role | Upstream path class |
| --- | --- | --- |
| gfx1100 | XTX 0/1 | CPU-rooted PCIe paths `00:01.0` / `00:01.1` |
| gfx1201 | R9700 | CPU-rooted PCIe path `00:06.0` |
| gfx1030 | 6900 XT | chipset/PCH path `00:1d.0` |

The raw PCI tree showed separate switch paths for each GPU and the
chipset-routed path for gfx1030. Device ordinals, UUIDs, and bus addresses are
diagnostic observations only; they are not used as the persistent topology
identity.

## Raw evidence hashes

The complete host-local snapshot remains in the Brutus evidence directory
created for this run. These hashes bind the observation without copying
machine-local raw output into the repository:

| Evidence | SHA-256 |
| --- | --- |
| GPU inventory | `88a9780c8f09eb3abdc616a4e44bfc090f5b8197bea91b8ea97188623ae9fefc` |
| HIP configuration | `578bb396c68ad3431fb20459bb8f16b71fcac28cd7d529e65895c3d0beb1caef` |
| RCCL library hash record | `a9f374e74f707e685a76098d85e20bf043fe1097a27f2ad58f0963c75517861c` |
| RCCL build ID record | `28204fb8d2675ed0efafa957fae005725b73d6ca343910ae8e30955834a4b1ae` |
| RCCL version strings | `b6c40f3d5b78f7076cfe8bdcda87275dd95b6d89e4886bef838050cdc4e21a14` |
| PCI tree | `dcdcfece3df33ac23f37de1b1e5a5593986412ec0f56968bbcdb1953d29c5fc6` |
| Full PCI detail | `719d27869ec9f82f9190cbce3640238374de57a374ae128ddf2390afdb17b402` |

## Current HIP AtomicOps capability probe

On 2026-09-10, a temporary HIP probe was compiled and executed on Brutus
using `/opt/rocm-7.2.4/bin/hipcc`. It queried
`hipDeviceAttributeHostNativeAtomicSupported` for every visible device and
reported successful HIP calls for all four devices:

| Device | Result |
| --- | --- |
| 0 | `value=1` |
| 1 | `value=1` |
| 2 | `value=1` |
| 3 | `value=0` |

The temporary source hash is
`061e9373d85631c5d9818a1cdfa6e5d6b033626a55c60608da989d095024f7cf` and the
probe binary hash is
`04043b62ad86e16b84907e7e49898685bc92cde6fc340482b37300de597dd34a`.
This confirms the current shared predicate's observed per-device boundary,
but it is not complete PCIe-component AtomicOps/transport evidence and does
not by itself authorize a topology allowlist.

## Crash-isolated direct RCCL qualification

A bounded current-code campaign was run on Brutus against the exact observed
RCCL library identity, using `all_reduce_perf`, 30,720 float elements, Ring /
Simple, and three attempts per topology. The campaign recorded 15 cases in
`cases.jsonl` (SHA-256
`095daced9c33197c2690cb17340781135509b2e0177915de79cd11c447d2ab8a`):

| Topology | Result |
| --- | --- |
| XTX + XTX (`0,1`) | 6 `pass` (including three post-fault control rechecks) |
| XTX0 + R9700 (`0,2`) | 3 `pass` |
| XTX1 + R9700 (`1,2`) | 3 `pass` |
| XTX0 + 6900 XT (`0,3`) | 3 `gpu_fault` |

Each case ran in a separate child process; after every negative-control fault,
the homogeneous control passed before the campaign proceeded. This is direct
RCCL-tests evidence only. It is not a managed BigCherry dispatch or 0840
secondary-communicator result, and the topology identity is intentionally
incomplete, so these observations do not enable patch 1225 admission.

The earlier qualification artifacts remain authoritative historical inputs:

- `rq04-01/environment.txt`:
  `36d91dfad7870aba539b68e146026da137c752e4aa2172ce16e259698a2cffa8`;
- `rq04-01/rccl-target-controls.txt`:
  `8135fe200e25bdc3a5f785470efe60d23062b162ca31e1afe802044cd4d20f0e`;
- `rq04-01/rccl-code-objects.txt`:
  `a87eb26585998a0f679b548ad767ce376818a6e143e1198cb3bbe362a2380ec5`.

## Disposition

This snapshot establishes useful placement and runtime provenance, but it does
not yet contain complete AtomicOps/transport evidence for every PCIe endpoint
and upstream component, nor managed direct and 0840 secondary-communicator
round-trip results bound to this exact identity. Therefore:

- `RcclTopologyEvidence` can serialize this class of evidence but remains
  incomplete until those fields are captured;
- patch 1225 continues to fail closed for unknown/unqualified records;
- no positive topology allowlist or PHA04 candidate binding is justified.

## Reproduction

The raw snapshot was collected with read-only `rocm-smi`, `hipconfig`,
`ldconfig`, `readelf`, `strings`, `lspci`, `uname`, and module-version probes.
The exact command output and per-device PCI details remain in the environment's
evidence directory and are referenced by the hashes above.
