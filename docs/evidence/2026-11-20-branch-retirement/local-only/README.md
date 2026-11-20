# Local-only branch preservation

Historical binary-safe patches plus precise original branch tips and SHA-256s are recorded in [inventory.json](inventory.json). These patches are **not** production changes. Recover only at the recorded historical merge base.

- node/radiance-main-integration: superseded by current Radiance 2.2 migration; no distinct patches relative to latest-bump at initial review, but its original full diff remains here.
- node/radiance-22-candidate: older 2.1/2.2 fixtures and ROCm workaround; must be reviewed before any retirement.
- feat/iq6r-v4-qualification: ongoing IQ6R experimentation; protected active branch, source copied as checkpoint only.

All source worktrees remain untouched. Nothing here establishes GPU correctness, speedups, or release qualification.
