# TO03 shared-worktree concurrency exercise

Observed 2026-09-08 UTC on `brutus` in a fresh temporary detached worktree at
revision `52ab88bebc74e800645f0694327f3a8bf4965195`; the dirty canonical
checkout was not modified.

Two separate shell processes used the same checkout. Session A created and
held probe A. Session B observed A as pre-existing, staged only probe B,
verified probe A was not staged, removed only B, and signalled. Session A
then verified its bytes were unchanged, removed only A, and asserted a clean
checkout. The final result is `PASS`. Transcripts were captured externally and
copied here only after cleanup.
