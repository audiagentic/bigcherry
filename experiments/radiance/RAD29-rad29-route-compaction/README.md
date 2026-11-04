# RAD29 experiment: rad29-route-compaction

Archived from branch `feat/rad29-route-compaction` (tip `ba54b47fba43471b5a1b78795fb82c03487db114`, base `a892e0cb20a58c1e3f08d9b549007e8d72918805`). Not part of any build: the code is kept as an inert patch.
Plan item: RAD29 (see `EXPERIMENT-REGISTER.md` for outcome and reason).

Revisit: `git switch -c exp/rad29-route-compaction <current main> && git apply --3way experiments/radiance/RAD29-rad29-route-compaction/change.patch`.
It may need rebasing: engine files have moved since the base commit.

Commits on the branch:

- 2026-10-28 ba54b47fb test(radiance): RAD29 compaction in route_compact.h, equivalence + timing test
- 2026-10-28 0692ee049 perf(radiance): RAD29 wave-aggregated route compaction in mailbox publish
