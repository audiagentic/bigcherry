---
id: BCOP41
order: 41
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T14:09:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Prefer native MTP confidence gating before custom adaptive depth

## Disposition

Authoritative owner: PRBE52.

Subsequent BigCherry work has already reconciled and hardware-tested 1255/1268 on Flash-Next. The custom adaptive controller lost to production fixed depth 3 at both recorded test depths and is rejected for that deployment.

Independent commit be8659acc88ad00ec6cffaf082433b98b31ccb97 added a bounded screen of llama.cpp's existing spec-draft-p-min confidence gate. Run that existing screen before any WHIRL or entropy-controller work.

Accept native confidence gating only if it improves median effective decode throughput by at least 5% over fixed depth 3 at both representative depths, has no held-out regression over 2%, remains repeatable under identical settings, and introduces no runtime failure. Otherwise retain fixed depth 3 and close Flash-Next adaptive-depth work.

Upstream entropy gating is assessment-only. Do not add another runtime confidence mechanism unless native p_min first passes and a separate held-out workload demonstrates at least 5% residual opportunity.

Dependencies: existing queue-pmin-screen.sh and production Flash-Next binary. FMTP remains separate and paused.
