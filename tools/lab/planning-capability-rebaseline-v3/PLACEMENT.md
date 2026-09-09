# Repository placement

Preferred repository location:

`tools/lab/planning-capability-rebaseline-v3/`

This follows BigCherry's rule that plan-specific/exploratory tooling starts under `tools/lab/<plan-topic>/` and remains outside the production package.

Before committing the Python files:

1. Apply the semantic rows in `TOOL_DISPOSITION_SNIPPET.md` to the current `docs/reference/tooling/TOOL_DISPOSITION.md` (the patch intentionally avoids hard-coding the prose row count; reconcile any count text with the actual current registry).
2. Confirm the three script paths have exactly one registry disposition.
3. Run `PYTHONPATH=tools python -m bigcherry check --quick`.
4. Do not add `tools/lab/__init__.py` or import these scripts from production/tests.
5. Generated outputs stay under `artifacts/lab/planning-capability-rebaseline-v3/` and are not evidence authority.

The pack itself does not create/update/delete plan items. Plan lifecycle operations remain owned by `ag-planning`.
