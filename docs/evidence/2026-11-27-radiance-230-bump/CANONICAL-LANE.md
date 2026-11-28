# Canonical lane reuse and production syntax qualification

- Source: Radiance 2.3.0, pin `3b13fd32663ef52252a7f0be255523480aa22ffe`; production patch composition and native units from controller `1591f7029`.
- SDK: recipe-selected ROCm 7.2.4 (AMD Clang 22).
- Full production `patch-compose-check --source radiance --compile syntax --json`: **504/504 host/device syntax passes**, gfx1100 and gfx1201; no skips, failures, or gate-1 issues. See [production-syntax.json](production-syntax.json).
- The prior generic LLVM 23 failure is not release evidence; it was corrected by selecting the named recipe SDK.

The previous bump compile gate configured a scratch build with the union of GPU targets, built the runtime and all components, and then independently invoked the canonical lane for CPU smoke. The normal bump path now supplies a lazy canonical-build callback to the component gate after composition anchors and syntax pass. The planner runs once, and ABI checks and offline component tests consume its runtime home. The lane phase records the existing build plan and verifies the launcher SHA256 before smoke, rejecting a missing or changed artifact. Standalone/explicit override paths retain the original component gate.

Regression suite: `test_bump_gates`, `test_pin_bump`, `test_pin_bump_resume_inputs` — 130 tests passed. New cases exercise canonical artifact reuse, ABI rejection, missing runtime, and a single build across the compile and CPU-smoke phases. Initial new fixture lacked a mocked upstream drift gate; corrected the synthetic fixture before the complete rerun.

**Scope:** syntax qualification and workflow tests only; this is not a linked runtime, GPU numerical/quality or model-serving qualification. The canonical build still must run, followed by component ABI/CTest, CPU smoke, queued GPU differential and model acceptance. No release or deployment approved.
