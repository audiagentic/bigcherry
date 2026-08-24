# Patch validation

Validation is fail-closed. Required capabilities need required producers;
missing, stale, tampered, or fabricated evidence is not PASS. Every artifact
is path-contained and SHA-256 bound to the campaign identity.

Validator outcomes are `PASS`, `FAIL`, `BLOCKED`, or `ERROR`. `BLOCKED` means
an external prerequisite such as required hardware is unavailable; it must not
be converted to PASS. Activation claims use a positive trace plus a disabled
negative control. Performance claims require causal activation evidence.

Custom validators implement exactly `check(ctx)`. Built-in dispatch is
immutable. `bigcherry check` is local and non-mutating; hardware validation is
performed only by an explicit campaign.
