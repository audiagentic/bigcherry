# RCD planning validation harness

Purpose: falsify the cross-cutting assumptions in `docs/design/JOBS_ORCHESTRATOR.md` before production implementation. This directory is lab-only; production code belongs under `tools/bigcherry/jobs/` and `tools/bigcherry/hardware/`.

Run:

```bash
PYTHONPATH=tools python tools/lab/run-campaign-durability/mock_pipeline.py --self-test
```

Expected:

```json
{"checks": 25, "ok": true}
```

Covered without hardware:

- capability-based GPU resolution by architecture/count/VRAM;
- peer-pair and exact-device selection;
- detection that subset/topology constraints require Slurm over-allocation rather than per-slot GRES types;
- allocation visibility may narrow an already-exclusive allocation but cannot broaden it;
- `BIGCHERRY_GPU_CLAIM` UUID/architecture grammar and fail-closed malformed/unknown claims;
- dynamic production conflict policy and fail-closed unknown claims;
- Linux ROCm vs Windows HIP environment-hash separation;
- hardware-cohort identity: locator move is operational drift, topology change/replacement changes the scientific cohort;
- inventory drift classes for locator/topology/device-set change;
- exit-code retry semantics (`75` same-commit requeue, `76` new attempt, `77` block);
- FakeExecutor dependency ordering for prepare -> execute.

Not covered and must remain hardware acceptance gates:

- actual AMD UUID/RSMI unique-id availability and stability;
- generated `gres.conf` accepted by `slurmd -G`;
- cgroup `/dev/kfd` + render-node behavior for both ROCm toolchains;
- HIP/ROCr enumeration under cgroups;
- peer access and `-sm tensor` on the installed cards;
- llama-swap claim/process attribution against real production processes;
- scheduler isolation/noise equivalence;
- systemd/sudoers/Slurm behavior on Brutus.

The simulator intentionally contains no Slurm, ROCm, Windows HIP, llama-swap, repository, or evidence-writing imports; it validates domain boundaries rather than pretending to validate hardware integration.
