# RSC11 S13(b) — staged v2 serve supervisor lifecycle

The **one active production serving contract remains v1** until the atomic
S13(c) engine/helper/manifest/launcher cutover. This slice wires the
`bigcherry serve` CLI through the existing typed S11/S12/P3 lifecycle
interface and **deliberately refuses live start** until that single activation
installs genuine `V2Bindings`: a native P1 measurement source, strict S10
manifest writer, and S5 engine pre-weight validator. It never falls back to
a second v1/v2 launcher or treats a test fake as a production provider.

`bigcherry serve start NAME --run-id ID` accepts explicit operator timing
intervals `--readiness-s`, `--terminate-s`, `--poll-s` and
`--stop-wait-s`. There are no hidden lifecycle timeout constants. The
staged `start_v2_with_bindings` uses **exactly one**
`start_with_adapters` → `MeasuredPlanValidator` → `Supervisor.execute`
path, with the same resolver as `serve explain`, the same S12 power
journal directory, the strict v2 manifest check and native process handles.

An explicit `--power-profile NAME` must match the named manifest profile;
it does **not** trigger hardware changes. When the S10 v2 manifest's resolved
power mode is `observe`, all requested settings must be empty, irrespective
of any profile selection. When mode is `apply`, startup refuses unless:
a non-stock profile is explicitly named; its manifest digest/profile match;
the trusted S10 qualification loader returns a typed proof for the **exact**
BDF → settings set; each writing BDF has an evidence identity; and the
proof binds engine/helper builds and the driver. No qualification loader is
installed pre-cutover, so power apply cannot occur in production. Once
activated, only the S12 supervisor may apply and restore power, and only
after verified child exits.

`bigcherry serve stop ID --poll-s N --stop-wait-s N` requests shutdown
through the P3 run-record/stop-request protocol. It accepts only a durable
run record identifying a `bigcherry/v2` manifest and power-receipt location.
It never signals stored PIDs or restores power from the stopping process,
waits for the lease-owning supervisor's terminal clean record, and refuses
pending or unverified exit. The run may be inspected through `serve status`
without claiming a process is alive merely because an older record says
`running`.

**Blocker for live activation:** The actual native full P1 record and
qualified power/owner-map source are not installed; the v2 S10 writer and
engine/helper readers still do not form a single production contract. The
CLI's `production_v2_bindings()` intentionally fails closed. This is not
a second active v2 path. Hardware S8 evidence remains lead-owned.
