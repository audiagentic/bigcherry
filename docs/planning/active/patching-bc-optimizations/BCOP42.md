---
id: BCOP42
order: 42
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-06T22:08:00+11:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: S
---

# Gate MET04 placement on permutation attribution

## Disposition

Authoritative owner: MET04 / patch 1283.

Subsequent BigCherry work has already implemented static usage-based expert reordering and measured the owner's XTX-dense / three-GPU expert layout. The same-file EP result is +3.5..4% decode at n=2, but the reordered row-split control is itself ~2 t/s slower than the original file, produces different output, and changes MTP acceptance. The remaining question is therefore attribution, not another placement mechanism.

Before 2:2:1 weighting or 245K qualification, require identity-rewrite and permutation/inverse byte controls, then original-vs-reordered row-split greedy/MTP-off inference, followed by same-reordered-file row-split-vs-EP MTP-off ABBA. Promote further placement work only with >=3% repeated MTP-off decode gain and non-overlapping 95% CI; otherwise retain production row split and terminate MET04 tuning for this model.

Do not create another scheduler, placement registry or cache. MET01 owns route/residency evidence; MET02/1281 range execution; MET04/1283 static placement. Treat single-trace routing-balance scores as in-sample until workload-separated traces exist.
