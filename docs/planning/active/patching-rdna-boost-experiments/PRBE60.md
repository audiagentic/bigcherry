---
id: PRBE60
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:57:41.013607+00:00'
breadth: ''
skill: intermediate
created-by: capability-rebaseline-v3
work: S
priority: null
---

# SYS-PCIE-001: PCIe ASPM performance policy

## Description

TODO (host/OS tuning, not code). Compare Linux PCIe ASPM 'performance' vs 'default' policy for bandwidth-bound discrete RDNA hosts (R9700/XTX). This is operations documentation, not a llama.cpp/BigCherry code patch -- no source changes, no runtime toggle. Relevance at b11126: N/A (not a source-code item); ASPM is a kernel/BIOS-level PCIe link-power setting outside llama.cpp entirely.

NON-PATCH ITEM, reclassified per GPT review as host-ops/tuning documentation, not a BigCherry patching-plan item in the code-implementation sense (this plan already correctly scoped it that way -- GPT's remaining defect was the pcie_aspm=force boot-option caveat). No BigCherry patch package, no llama.cpp source change, ever. GPT correction: `pcie_aspm=force` is a KERNEL BOOT PARAMETER (requires reboot, changes ASPM enforcement mode system-wide at boot) and is materially different from setting the runtime ASPM policy via `/sys/module/pcie_aspm/parameters/policy` -- the ordinary measurement procedure must not casually invoke `pcie_aspm=force`; it is a separately-approved, higher-risk platform experiment only.

## Steps

1. Record current host ASPM policy: `cat /sys/module/pcie_aspm/parameters/policy` and per-device link state `lspci -vvv -s <bdf> | grep -i aspm` for each GPU on Brutus/target hosts.
2. Capture cold-boot baseline llama-bench PP/TG for a dense (Qwen3 dense Q4) and MoE (Qwen3 MoE Q4) model at default ASPM policy, GPU fully resident (no CPU offload) and a transfer-heavy control (partial CPU offload / large batch prefill).
3. Set ASPM policy to 'performance' (`echo performance > /sys/module/pcie_aspm/parameters/policy`, requires root; on some kernels needs `pcie_aspm=force` boot param plus a fresh boot), repeat identical benches.
4. Record PCIe link state (`lspci -vvv | grep -i 'LnkSta'`) and idle power draw (`rocm-smi --showpower` or platform equivalent) for both policies.
5. Compare TG/PP deltas, transaction-latency proxy (time-to-first-token under contention), and idle power cost across >=3 cold boots per policy to average out boot-to-boot noise.
6. Write findings to docs/ as host tuning guidance (not a patch): state policy recommendation is local-hardware-specific, include exact commands and measured deltas, and explicitly say no llama.cpp runtime change was made.
7. If no repeatable benefit is found, close this item as informational-only with 'no measurable local benefit' recorded.

1. Record current host ASPM policy: `cat /sys/module/pcie_aspm/parameters/policy` and per-device link state `lspci -vvv -s <bdf> | grep -i aspm` for each GPU on Brutus/target hosts, BEFORE any change.
2. Capture cold-boot baseline llama-bench PP/TG for a dense (Qwen3 dense Q4) and MoE (Qwen3 MoE Q4) model at default ASPM policy.
3. Set ASPM policy to 'performance' via the RUNTIME sysfs knob only (`echo performance > /sys/module/pcie_aspm/parameters/policy`, requires root) -- do NOT add `pcie_aspm=force` to the ordinary procedure; if the sysfs knob has no effect because the kernel/BIOS locked ASPM off entirely, record that as the result ("no measurable local benefit / policy not controllable at runtime on this host") rather than escalating to the boot parameter.
4. Repeat identical benches under the runtime-only policy change.
5. Record PCIe link state (`lspci -vvv | grep -i 'LnkSta'`) and idle power draw (`rocm-smi --showpower` or platform equivalent) for both policies.
6. Compare TG/PP deltas, transaction-latency proxy, and idle power cost across >=3 cold boots per policy.
7. Write findings to docs/tuning/pcie-aspm-policy.md: state the recommendation is local-hardware-specific, include exact commands (runtime sysfs only) and measured deltas. If `pcie_aspm=force` is ever separately, explicitly approved as its own platform experiment (different item, different sign-off), document it there -- never fold it into this item's default procedure.
8. If no repeatable benefit is found via the runtime knob, close as informational-only with 'no measurable local benefit' recorded.

## Detailed Solution & Technical Design

Out-of-repo, OS-level operations guidance. No dispatch, no gating, no source anchors -- this never becomes a BigCherry patch package. The only 'implementation' is a reproducible measurement procedure and a documentation artifact. Do not add a runtime ASPM toggle to llama.cpp/BigCherry under any circumstance (explicit acceptance-criteria prohibition in the item).

## Code Samples & Guidance

No code changes. Reference commands only (see steps).

## Files

docs/tuning/pcie-aspm-policy.md (new, host tuning notes + measured results); no llama.cpp or patches/ files touched.

## Validation

No model-output impact expected (link-power policy does not change compute results) -- confirm with a quick temp-0 identity spot check as a sanity control. Primary validation is the reproducibility of the TG/PP/latency/power deltas across >=3 cold boots per policy on the actual target host (R9700/XTX). Hardware runs happen on Brutus; this agent does not run them.

## Effort & Risk

S / low risk -- no code path, reversible OS setting, worst case is a reboot to restore default policy.

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Document as optional host tuning only if local hardware reproduces a repeatable benefit; never add a runtime ASPM toggle or universal recommendation.

## Notes

Supersedes: RD77
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd77

append

2026-09-24 relevance at b11126: N/A, host/OS ASPM tuning item, no llama.cpp source touched. GPT design request: not applicable (no code target).

2026-09-24 GPT review req_2b65d50ebe9547fd applied: NOT-READY -- removed pcie_aspm=force from the ordinary procedure (it is a separate, higher-risk kernel boot parameter, not the runtime sysfs policy knob); documented it only as a separately-approved platform experiment if ever needed.

## Change Log

- 2026-09-09T10:57:41.013607+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:14:51.599098+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.398077+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.212166+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:15:55.582608+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.945234+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:25:47.906961+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
- 2026-09-24T02:26:19.310074+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:43:33.105346+00:00 (updated-by): Updated: section:description, section:steps
- 2026-09-24T04:43:42.777950+00:00 (updated-by): Updated: section:notes
