# BigCherry tools

Normative tooling guide: [`docs/reference/TOOLING.md`](../docs/reference/TOOLING.md).

- Pin/release: `PYTHONPATH=tools python -m bigcherry repin` / `pin-status`
- Source/build: `audit`, `generate`, `build`
- Patches: `patches`, `patch-lint`, `patch-validate`
- Campaigns: canonical campaign build/lane APIs
- Tuning/replay: journal, promotion, replay and catalog commands
- Experiment Contracts: `experiment-contract`
- Diagnostics: `check`, `doctor`
- Offline analysis: intended destination `bigcherry.analysis`
- Experiments: `tools/lab/<plan-topic>`
- ROCm setup: `tools/env/`

Search before creating a second framework. Patch-specific proof belongs with the patch package; uncertain exploratory work starts in lab.
