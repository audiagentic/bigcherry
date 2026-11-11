# RSC11 S13a — native live-room measurement integration (inactive)

`bc-serve --publish-v2-live-room --site SITE [--profile PROFILE] [--set KEY=VALUE]
[--sys-root DIR]` is a new **read-only** command. It detects the current
KFD/PCI inventory, resolves selected engine/helper BDFs, reads live host
MemAvailable and effective memory-cgroup room, and measures AMD global
pinnable-room capacity using the driver's cap and the current PCI GTT usage
counters. It outputs `bc_serve_v2_live_room` with three byte fields and the
selected BDFs. It does **not** write either manifest version, launch any
process, or apply power.

The reader fails closed when AMD driver/cap state, selected BDF inventory,
memory-cgroup room, or GTT counters are missing or malformed. It subtracts
GTT usage for **all visible adapters**, not only the selected layout, before
publishing remaining cap room; the result is additionally bounded by live
host/cgroup room. The driver may account other kinds of long-term pins
outside these GTT counters; this publication is *provisional* and is not
a qualified P1 `pin_available_bytes` input for production S8 admission until
a matching driver-level pin ledger is verified on hardware. Its limitations
are retained in `provenance`. Do not use the number as a guaranteed
reservation or set a margin from it.

This is not a complete P1 measured-input publication. The existing
`v2/p1_measurement.cpp` requires real rad0011 engine bytes, original
owner-map identity, per-rank required/fixed GPU bytes, worker footprint,
model unowned experts and measured resident-other bytes. The current
production engine plan records an aggregate `PlanVram` and some per-card
facts, but does not publish the exact per-rank fixed/required budget as a
single typed sidecar; the owner map currently arises inside the v1 manifest
writer. Neither value may be reconstructed from historical v1 pool/host-KV
defaults or copied from a v1 manifest. Until authoritative producers are
connected, `P1MeasurementSource` remains inactive for production and
`bigcherry serve start` remains refused. The final S13 switch must consume
only one strictly authenticated v2 record, with qualified P2 store records.
