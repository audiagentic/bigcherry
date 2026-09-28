---
id: PVPS11
order: 0
plan: patching-validation-package-standard
state: pending
created-at: '2026-09-26T10:47:05.216165+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: L
---

# Vulkan-backend validation campaigns (build, attest and benchmark Vulkan patches)

## Description

The patch validation campaign (validation_campaign / scaffold / build.py) only builds and attests HIP. 1269_prbe55_vk_smalln_dmmv (first backend="vulkan" patch, rebase CLEAN 2026-09-26) therefore cannot be validated. Add a Vulkan lane: GGML_VULKAN builds of the four arms, RADV device attestation (vendor/device/driver + PCI locator), llama-bench/llama-server lanes on Vulkan devices, and test-backend-ops -b Vulkan0 correctness.

## Steps

1. Build: backend-typed build plan in campaign/build.py (GGML_VULKAN=ON, no HIP targets), keyed build roots per backend+driver.
2. Attestation: parse Vulkan device lines (ggml_vulkan: device name, driver, uuid/PCI) into ExecutionIdentity(backend="Vulkan"), map GGML_VK_VISIBLE_DEVICES to locators from environment.local.toml.
3. Scaffold: select arms/binaries by patch backend; device-map uses Vulkan ordinals.
4. Producers: support.select_device/device_env for Vulkan.
5. Queue: run_campaign.sh passes --backend vulkan; re-validate 1269 on gfx1100 and gfx1201 (RADV).

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

Unit tests for the Vulkan attestation parser and build-plan selection; an end-to-end campaign of 1269 producing 4 sessions per architecture.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

Owner 2026-09-26: 'need to actually test the others properly' - covers 1269, which has no validation path today. Framework work (no promotion gate).

## Change Log

- 2026-09-26T10:47:05.216165+00:00 (created-by): Created by agent
