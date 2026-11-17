# RCS13 / PR #946 — Run 9 source-isolation stop after VRAM barrier

**Verdict: NOT CLEARED; zero valid A/B/B/A serving samples.**

The PR #946 candidate source `17a16f5d71` is seven-file-isolated
against main `7c0273976`, and the new full Radiance build passed ABI 35/35
and 30 validated / 0 pending patches. It has the same upstream slice,
Radiance patch recipe, effective CMake (apart from the equivalent
ccache launcher paths), engine and libr4d bytes as the retained A build.
See `build-attestations.json`.

The **candidate's production strict-v2 measured manifest now passes** with the
same current-main Python planner as planned for both arms (SHA-256
`cd09429a7c7a141734b60009074287a87e272e921a2e23ec33220ea85fc21397`), containing the BDF VRAM-release barrier. Actual imported
planner was printed in the invocation log; the trace shows **two**
fresh measuring helper starts and host-pool remeasurement from
**32005 to 32301 MiB**. The final manifest has **12641** expert owners,
owner map `744477bc8f39f18bdc68f472b4f83d6cbb2b576c0095439fe463cd84427ca391`, and is bound to candidate's
own bc-serve producer. GPU lease released cleanly.

**New SOURCE ISOLATION gate failure (CPU-only):** the retained control A was
built from `e28734f097`, while current B is built from
`17a16f5d71`. There are **22 changed files** between the
actual source commits, not seven. The 15 extra files include C++
production serving `engines/radiance/components/serving/session.cpp`,
`runtime_guards.cpp` and `calibration_lab.cpp`, merged with PR #1059
between the two builds. Relevant `session.cpp` hunks include lines
741–745, 1403–1404 and 1838–1841. The built `bc-serve` SHA-256
differs across arms (as shown by the binary attestations). Therefore the
same-seven-file branch diff does **not** make the old retained A build a
controlled comparison for the newer B binary set.

The portable `cpu_source_gate.py` reproduces the mismatch without GPU,
model, credentials or other host-specific inputs:

```sh
python3 cpu_source_gate.py --repo /path/to/bigcherry-checkout
```

As instructed for a new gate failure, Run 9 **stopped before A/B/B/A**.
No decode engine ms/step, prefill/TTFT, MTP draft/accept counts,
output parity, executed K=2560 W4H2 dynamic-LDS kernel counts or occupancy
can be claimed. The production candidate manifest result is useful, but
it is not a performance comparison. A fresh current-main A build (or a
candidate built against the exact A source) is required before proceeding.

No commits were made to the PR #946 branch; this evidence is docs-only.
