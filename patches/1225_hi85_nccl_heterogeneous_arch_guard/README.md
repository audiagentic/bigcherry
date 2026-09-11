# HI85 heterogeneous-architecture guard

This package contains the optional NCCL/RCCL heterogeneous-architecture
fail-closed guard (HI85). The durable contract is: mixed-architecture
communicator initialization must reject before launch; homogeneous and
non-NCCL paths remain unchanged. Architecture rationale and profiling context
are summarized in `docs/reference/architecture/MULTI_GPU_DISPATCH.md`; patch
identity and lifecycle metadata are authoritative in `patch.toml`. Real
hardware evidence and the GP02 rewrite history are recorded in
`SUMMARY.md`'s "Real hardware validation" section.

## Lifecycle: promoted to `validated` (2026-09-11)

`state` was `"untested"` despite GP02's own plan item being marked
`completed` with real hardware evidence covering both the fail-closed
rejection path and the successful-admission/no-false-reject path (see
SUMMARY.md): `{0,3}` correctly declined RCCL and fell back to a working
internal-pipeline inference run (pp512=429.51, tg8=23.11 t/s); `{0,2}` and
`{1,2}` correctly admitted RCCL with real RCCL-speed numbers (pp512=1272.84
and 1318.29 respectively), confirming no false-positive-reject regression
on topologies that must remain admitted; the full 5-patch chain
(0100+0830+1001+1225+0840) applied and built cleanly and idempotently from
a clean pinned checkout.

Requested an explicit GPT solution-approval decision on whether this
evidence supports promotion given that a separate, broader
production-qualification gate (a 20-fresh-process repetition requirement,
owned by GP01/GP06/GP07's own qualification work, not this patch's scope)
remains open elsewhere. **Verdict: PROMOTE** (dev-gpt-agent, session
`ses_f860c44236da4101`, `req_9fbad4bbc45c4bde`): "GP02's patch-local safety
claim is directly qualified by real hardware covering both fail-closed
rejection and successful admission/no-false-reject paths; the
20-fresh-process requirement is a separate production-qualification gate
... and should not block this patch's own lifecycle promotion."

`patch.py`'s `STATE`, `patch.toml`'s `state`, and `SUMMARY.md`'s `Status`
all corrected to `"validated"` per this verdict.

## Known limitations

- No `validation.toml` adapter exists; this patch has no
  `BIGCHERRY_PATCH_HIT`-style trace marker, so no activation check is
  wired. Evidence above is real-hardware functional confirmation
  (single-rep across 3 real topologies), not the 20-fresh-process
  repetition rigor a full production qualification would use -- that
  broader gate belongs to GP01/GP06/GP07, not this patch.
- Patch 0840 (`hybrid_allreduce_dispatch`) hard-requires this patch
  (`requires = ["1225_hi85_nccl_heterogeneous_arch_guard"]`) specifically
  because it brings up its own independent `ncclCommInitAll()` call site
  that this guard's shared admission predicate protects.
