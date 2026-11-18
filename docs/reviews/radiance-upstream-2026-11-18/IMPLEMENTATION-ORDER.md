# Implementation queue

The work packets are for implementation after this planning handoff. Keep each patch small enough to review and revert. Take a fresh task worktree for each independent packet and keep the accepted source revision and evidence with that packet.

| Lane | Start condition | Order and stop point |
| --- | --- | --- |
| Build cap | No migration dependency | [BU02](../../planning/active/build-ci-speed/BU02.md)'s fake-runner slice first. Run cold paired CI only after all four calling paths honor the same cap. |
| Upstream bump | [RUM01](../../planning/active/patching-radiance-upstream/RUM01.md) decisions and baseline approved | Follow the [migration handoff](IMPLEMENTATION-HANDOFF.md) and [execution guide](EXECUTION-GUIDE.md). Run RUM03, then RUM04, then extract only survivors in RUM02. Keep RUM10 out of this release. Advance only on each packet's acceptance. |
| Performance | Establish trustworthy benchmark artifact identity before measuring gains | See [BRBC05](../../planning/active/build-reusable-build-campaign/BRBC05.md). See [RAD54](../../planning/active/patching-radiance-rdna3/RAD54.md). See [RSC12](../../planning/active/run-serving-configuration/RSC12.md). Follow the [optimization handoff](../astra-performance-2026-11-18/IMPLEMENTATION-HANDOFF.md). Keep warm replay and MoE independent. Obtain a source review of the synchronization change before the hardware gate. |

Use existing fixture targets and receipts. Escalate a mismatch in manifest identity, ABI, physical device order, ticket retirement or numerical output before changing the accepted design or broadening the test matrix. Stop after the first unexplained mismatch and record the reproduction. Scheduled implementation starts only when assigned; this planning pass makes no runtime change.
