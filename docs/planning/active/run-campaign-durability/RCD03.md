---
id: RCD03
order: 3
plan: run-campaign-durability
state: pending
created-at: '2026-09-26T00:51:56.813138+00:00'
breadth: ''
skill: intermediate
created-by: agent
priority: P1
work: M
---

# Install and configure the job platform on Brutus (Slurm: munge, slurmctld/slurmd, AMD gres, licenses, cgroups)

## Description

Assuming RCD02 selects Slurm: single-node Slurm on Ubuntu 24.04 as systemd services. Static gres.conf for 2x gfx1100, gfx1201, gfx1030 (renderD nodes + /dev/kfd), licenses measure=1 (host-exclusive timed measurement) and build=N, cgroup device constraint, accounting (sacct) storage, JSON output. Requires owner sudo for package install.

## Steps

1. Owner installs slurm-wlm, munge (sudo).
2. slurm.conf/gres.conf/cgroup.conf generated from environment.local device inventory (host facts stay untracked).
3. Verify GPU visibility inside jobs matches PCI locators (attestation), including renumbering under cgroups.
4. Smoke: one single-GPU and one dual-gfx1100 job running llama-bench.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

sinfo/scontrol show correct gres; a job on gpu:gfx1100:2 attests both 7900 XTX PCI locators.

## Effort & Risk



## Standards



## Acceptance Criteria



## Notes

## Change Log

- 2026-09-26T00:51:56.813138+00:00 (created-by): Created by agent
