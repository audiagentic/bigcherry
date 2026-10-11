# 1276: adaptive AllReduce over the N=3 root path

## Scope

Composes 0840's adaptive AllReduce with 1244's three-rank root path: reductions below
`--allreduce-switch-bytes` use the internal root3 provider, larger ones RCCL.
`BIGCHERRY_AR_ROOT3_ROOT=0|1|2` picks the root rank (default 0). Transport and synchronization are
1244's; only the logical root/leaf indices are generalized.

## State

`untested`: no hardware claim. Needs N=3 full-stack evidence (PGC10) and an experiment contract before a
validation run can start; the validation adapter is added together with that contract.

## 2026-10-09 qualification boundary

PGC10 is the authoritative owner; this package remains **untested/default-off**. Explicit `adaptive` + 1244/1276 does not extend the dual-gfx1100 `auto` default to N=3. The 96 KiB logical-byte switch is followed by 1244's separate F32/size gate. The 80 KiB prompt-tail shape can select root3, whereas validated QFP01/1291 CPU-root (64 KiB F32 default) routes it to RCCL. No root3-vs-CPU-root matched E2E result exists. Reuse existing 1276 composition tests and PGC10's bounded three-provider gate; do not queue a new hardware campaign without a demonstrated CPU-root bottleneck. See BCOP97.
