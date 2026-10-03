---
id: PRBE63
order: 0
plan: patching-rdna-boost-experiments
state: superseded
created-at: '2026-09-09T10:57:54.429485+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# VK-DRV-003: Cooperative-matrix bypass on AMD proprietary Vulkan

## Description

UPSTREAM-ABSORBED. The narrow driver-scoped coopmat bypass this item asked for (avoid coopmat routes specifically on AMD's proprietary Vulkan driver, not architecture-wide) already exists in b11126's ggml/src/ggml-vulkan/ggml-vulkan.cpp: the coopmat warptile-table selection (`device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary`) and the large mul_mat tile gate (`device->mul_mat_l[i] = device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary`) are both driver-id scoped, not global, and not architecture-only. No BigCherry patch implements or needs to implement this; it is native upstream behavior at the pin.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Grep evidence only, no hardware run required for this disposition: `git -C work/upstream/llama.cpp.git grep -n "eAmdProprietary" b11126 -- ggml/src/ggml-vulkan/ggml-vulkan.cpp` returns the two guarded sites (ggml-vulkan.cpp lines ~1732 and ~4579).

## Effort & Risk



## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

No global coopmat policy; conclusions scoped to fingerprinted driver/architecture/operation/quant/signature; independent of PRVP01-RO22 qualification.

## Notes

Supersedes: RD80
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd80

append

2026-09-24 relevance at b11126: UPSTREAM-ABSORBED. ggml/src/ggml-vulkan/ggml-vulkan.cpp already gates coopmat warptile selection AND mul_mat_l on driver_id != vk::DriverId::eAmdProprietary (verified at lines ~1732, ~4579 via git show b11126:ggml/src/ggml-vulkan/ggml-vulkan.cpp). Mechanism is driver-scoped, not global/architecture-only, matching the item's own constraint. No remaining code gap identified. GPT design request: gateway rejected all submission attempts this session (VAL-AGW-025 ownership-queue-error and EXT-GPTAUTO-003 composer-operation-timeout, visible in agent_task_gateway_overview) -- disposition made from direct source verification instead; no GPT request id.

2026-09-24 GPT review req_d55aed71224e43a8: READY

## Change Log

- 2026-09-09T10:57:54.429485+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:03.686191+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.411603+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.233203+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:16:32.506067+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_031644_repaired-four-more-active-succ_8062
- 2026-09-10T03:16:44.985999+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:29:50.584266+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:notes
- 2026-09-24T02:30:17.075898+00:00 (state-transition): State: pending → superseded
- 2026-09-24T02:30:32.419716+00:00 (updated-by): Updated: section:notes
- 2026-09-24T04:34:11.021047+00:00 (updated-by): Updated: section:notes
