# Deferred CI security hardening

Inactive archive of node/ci-pin-action-shas, [PR #1366](https://github.com/audiagentic/bigcherry/pull/1366), closed unmerged on 2026-11-22. Original commits: 078bf22bb51c515a0d341ea4c76acc23ddc3a568 (pin actions), 78cff34b0ca384d80d89d505e281a11df711af5d (permissions); tip 3d896bbaf98c3f8232c5c029d152d4ed86f38f6e, base 7adda17de08759cb09e9fdaa1a31fa5e8d6dcbd0.

Owner deferred this on 2026-11-27: security benefit does not currently justify maintenance and CI risk. No live workflow or mandatory test is activated. No fresh functional CI qualification is claimed.

Plans: [BU04](../../docs/planning/completed/build-ci-speed/BU04.md) for SHA pinning and [BU05](../../docs/planning/completed/build-ci-speed/BU05.md) for permission narrowing.

change.patch preserves the full delta from the recorded base; original-commits.patch preserves both original commit deltas (without email metadata). Historical ledger content is evidence only, not a new active event.

To reconstruct, use an isolated checkout of the recorded base and git apply --check followed by git apply change.patch. Reopen the relevant plan only on concrete security need or owner approval. Port selectively to current main, refresh trusted action SHAs, review permission inheritance, and verify ordinary PRs and real package publishing before enforcing any new gate.
