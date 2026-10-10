---
id: BCOP51
order: 51
plan: patching-bc-optimizations
state: deprecated
created-at: '2026-10-08T04:07:42+11:00'
created-by: agent
priority: P2
---

# PRBE07 BridgeSpec disposition

## Audit result

PRBE07/RD101 is complete as research triage. BridgeSpec 0.1.0 at `2b846f2ff1eb95ac84e4b0488882b7e4066bff14` is MIT-licensed and demonstrates useful AMD speculative mechanisms, but its shipped integration is a Windows/gfx1100/Qwen3.8-27B single-slot host-mediated sidecar with incomplete state lifecycle and no production Vulkan/HIP zero-copy synchronization.

## Authoritative owners

- PRBE07: terminal research record only.
- Native speculative state/verification: existing llama.cpp/BigCherry MTP owners.
- Cross-device draft overlap: FMTP.
- Verification-width MMVQ: existing BigCherry/upstream MMVQ qualification owners.
- Draft-vocabulary work: existing native draft-vocabulary owner.

## Subsequent work already acting on it

BigCherry now has native MTP, separate-draft-GPU scheduling/overlap work, draft-vocabulary work and width-sensitive MMVQ qualification. Importing BridgeSpec would duplicate those implementation paths rather than fill an unowned seam.

## Disposition

**Inspiration-only.** Do not import the sidecar ABI/runtime or create a new scheduler/state machine. Reuse only bounded mechanisms through their existing owners when current profiling proves >=5% wall-time opportunity and current-pin AMD qualification meets correctness plus CI95-low >=3% E2E / <=1% control-regression gates.

## Blockers / dependencies

No blocker remains for PRBE07 closure. BridgeSpec's published throughput is external single-XTX evidence and is not promotion evidence for BigCherry's Linux dual-XTX + R9700 + gfx1030/no-P2P topology. No build, test, prototype or hardware benchmark was run by this audit.

## Change Log

- 2026-10-10T10:25:46.203449+00:00 (state-transition): State: completed → deprecated
