# Serving measured-writer stop: baseline evidence

Baseline: `35744c47d`. The CPU-only Release serving test and its dependencies built successfully at low priority (`nice +19`, two build jobs). Repeating the complete `bc_serving_test` yielded **2 failures in 12 normal runs** and **1 failure in 12 CPU-loaded runs** (four competing busy loops on the test's four pinned cores).

All three local failures were the unrelated existing forced-placement negative-case report (`helper0 did not report its forced placement`), rather than the measured-writer cancellation reported by CI and release builds. These are full-suite rates, not an assertion that the specific writer race was observed locally.

Source analysis: the owner returns false from its event-driven planning wait on operator stop; `run_planning_child` verifies writer exit then throws `PlanFailed`. The plan callback boundary erases exception identity and the owner records a failed plan instead of a clean stop. The production fix should preserve a typed stop result through that boundary and retain verified cleanup, with no new timeout or retry.
