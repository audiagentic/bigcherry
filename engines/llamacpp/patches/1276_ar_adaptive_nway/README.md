# 1276: adaptive AllReduce over the N=3 root path

## Scope

Composes 0840's adaptive AllReduce with 1244's three-rank root path: reductions below
`--allreduce-switch-bytes` use the internal root3 provider, larger ones RCCL.
`BIGCHERRY_AR_ROOT3_ROOT=0|1|2` picks the root rank (default 0). Transport and synchronization are
1244's; only the logical root/leaf indices are generalized.

## State

`untested`: no hardware claim. Needs N=3 full-stack evidence (PGC10) and an experiment contract before a
validation run can start; the validation adapter is added together with that contract.
