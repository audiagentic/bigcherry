---
id: BCOP165
order: 165
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-12T15:10:17+00:00'
created-by: agent
priority: P2
work: S
---

# PVPS04: omit unused producer tune/replay builds only with v5 evidence

**Discovery:** Shared scaffold always builds tune/replay; producer uses them only as v4 identity fields, while legacy `validation_campaign.py` actually executes them. PVPS03 can add a distinct base build: saving is 2/5–2/6 build variants (33.3–40% by count), not measured time or token/s.

**Owner/consolidation:** PVPS04 owns existing scaffold's producer-only omission and schema-v5 provenance; PVPS03 retains ladder, PVPS01 multi-arm policy, PEC02 evidence completeness. No duplicate executor, fake build identity, or second schema registry. No independent PVPS04 implementation or experiment was found within 12 hours.

**Bounded action:** Explicit internal producer/tune-replay mode, exact v5 role sets, producer digest-v2 labels, unchanged legacy tune/replay execution and v1-v4 historical readback. Unit/mock tests must prove build counts, role/mode rejection, input digest, contract/ported-benched gates, old-record verification and ladder parity before any uncontended real producer run.

**Terminal gate:** ACCEPT only with zero false evidence PASS, preserved historical v4 verification, real producer stock/control/subject/optional-base builds and observed cold/warm build-time savings. Otherwise REJECT and retain existing scaffold. No hardware or repository build executed in this audit.

**Protected work:** Recent Radiance GDN/AllReduce/MoE/prefill, QFP15 qualification and PA44/release changes remain untouched.
