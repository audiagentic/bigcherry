# Patch validation

**Canonical reference:**
[`docs/reference/testing/PATCH_VALIDATION.md`](../testing/PATCH_VALIDATION.md)
— the full RD-patch validation methodology (package layout, when a
package is required, Experiment Contract binding, authoring
`validation.toml`/`README.md`, the real contract-execution architecture,
evidence storage, workflow, anti-patterns). This file exists only as a
short landing pointer from `docs/reference/patches/` — do not duplicate
policy text here; edit the canonical doc instead.

Quick facts worth repeating for anyone skimming this directory only:

- Validation is fail-closed. Required capabilities need required
  producers; missing, stale, tampered, or fabricated evidence is not
  PASS. Every artifact is path-contained and SHA-256 bound to the
  campaign identity.
- Validator outcomes are `PASS`, `FAIL`, `BLOCKED`, or `ERROR`. `BLOCKED`
  means an external prerequisite (e.g. required hardware) is
  unavailable; it must never be converted to PASS.
- Activation claims use a positive trace plus a disabled negative
  control. Performance claims require causal activation evidence, not
  merely a benchmark artifact existing.
- Custom validators implement exactly `check(ctx)`. Built-in dispatch is
  immutable. `bigcherry check` is local and non-mutating; hardware
  validation runs only via an explicit campaign
  (`bigcherry.patch.validation_campaign`).
