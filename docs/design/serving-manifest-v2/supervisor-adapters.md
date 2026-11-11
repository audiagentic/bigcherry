# P3 supervisor start/stop adapters (inactive until S13)

The active `bigcherry serve start|stop` router still refuses live execution.
Only S13 will atomically select the production v2 reader/validator and bind
`tools/bigcherry/cli/serve_v2.py` as the command handlers. This PR does not
alter v1 startup or call the existing Bash launcher.

The testable `serve_v2.start(args, source, writer, validator, limits)` binds
the canonical bc-serve layout resolver to the existing S11
`MeasuredPlanValidator` and S12 `Supervisor`. No alternative plan resolver
is implemented here. The S11 writer must deliver a strictly validated v2
manifest: P3 checks its wire identity and server health port, re-reads the
resolved per-BDF power section and **requires observe mode** (no device
power writes). A separately qualified apply path is an S13/S15 decision.

`SubprocessLauncher` uses direct argv execution, scrubs inherited hostile
`R3_*` variables and refuses those keys on new specs. Each helper must
announce its own readiness in its newly created log before the next helper
starts; the engine must pass its local manifest-selected HTTP `/health`
check. The launcher owns the original child process handles, sends TERM,
escalates to KILL only after an explicit configured grace period, and
requires `poll()`/wait exit evidence. Its readiness, stop and poll intervals
are **required explicit lifecycle policy**, not hidden hard-coded defaults.

`serve_v2.stop(args, limits)` sends an idempotent, typed stop request to the
foreground, lease-owning start supervisor; it never sends signals based on
persisted PIDs or restores GPU power from another process. It only reports
success once the supervisor's run record has reached `stopped` without
errors. Failed, missing, tampered, timed-out or unverified stops refuse.
S12 alone applies or restores power after verified process exits.

S13 must bind real engine-measurement producers (P1), selected scoped
calibration records (P2), canonical v2 command specifications, qualified
lifecycle deadlines, lease acquisition for the full start lifetime and
end-to-end canary/MTP guards. This PR's synthetic fake-process tests do not
prove live serving, GPU power restoration or hardware readiness.
