# RAD50 Gate 0 — offline handoff and overlap models

This is the independent, **card-free** feasibility slice of RAD50's CPU-to-GPU BAR-push investigation. It is lab-only, has no serving or vendor dependencies, does not allocate GPU memory, and cannot enable a new transport.

- `chunk_contract.py`: explicit CPU/GPU handoff state machine and negative cases for stale/duplicate tickets, premature publication, buffer reuse, and fault recovery.
- `critical_path_model.py`: illustrative overlapped CPU-producer/GPU-consumer makespan scenarios for mapped-host GPU-pull versus BAR-push, separating the CPU copy and direct-output variants.
- `test_chunk_contract.py`, `test_chunk_stress.py`, `test_critical_path_model.py`: deterministic offline regression tests.

Run from the repository root:
```sh
python3 -m unittest discover -s tools/lab/rad42-bar-push -p 'test_*.py'
python3 tools/lab/rad42-bar-push/critical_path_model.py --help
```

The default bandwidth examples are illustrative and some originate in earlier RAD40 measurements; all unmeasured stages must be replaced with real replay/profiling evidence. Passing these tests is **not** a GPU coherence, transport correctness, fault recovery, or end-to-end speedup proof.

Owner: RAD50 / Radiance remote-MoE. Disposition: experimental lab utilities; keep outside the installed BigCherry package. Subsequent RAD50 hardware probes and serving experiments are separately gated.
