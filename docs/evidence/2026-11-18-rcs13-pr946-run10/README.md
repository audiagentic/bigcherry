# RCS13 / PR #946 — Run 10 retained-bundle stop

**Verdict: NOT CLEARED.** Zero production A/B/B/A samples; no new decode,
prefill/TTFT, MTP draft/accept, greedy hash parity, or executed W4H2
dynamic-LDS activation evidence.

Source isolation passed: main `7c0273976` against candidate `17a16f5d7`
changes exactly the seven RCS13 files, not the 22-file Run 9 confound.

The prior candidate build `a0a99806f887aaecce91ac4369d4b17c`
was reported successful with matching ABI 35/35 and a strict-v2 measured
manifest (12,641 experts, host-pool remeasured from 32,005 to 32,301 MiB).
Its recorded manifest SHA-256 is
`37cadb185ea22578cf5fea526acebd48b49c59e133c2385d1e3d2c7d787deb02`.
Those are **historical Run 9 observations**, not fresh Run 10 measurements.

Run 9's recorded cleanup at **2026-11-17 14:24:36 UTC**, after evidence
PR #1063, deleted the candidate content-addressed bundle and associated
artifact-store/build-log directories. Run 10 began at **14:26:10 UTC**,
so the instruction to reuse and rehash that bundle could not be met.
A CPU-only probe found none of the 12 attested B members; the exact
candidate bundle could not be loaded. The full clean baseline A build
was started using the official build CLI, then its own detached process
group was terminated once this gate failed. Neither arm was used for
production A/B/B/A. No alternative baseline or rebuilt candidate was
silently substituted.

See `gate-and-lifecycle.json`, the path-sanitized 12-member
`candidate-attestation.json`, and the portable CPU reproducer.
The status is **artifact-retention stop**, not an observed RCS13
runtime regression. A subsequent authorized run must rebuild B,
rehash both arms and repeat the strict-v2 identity comparison.

No changes or commits to PR #946.
