# Tool disposition registry addition

Insert this section immediately before the existing `## HI168 retained investigation tools` section in `docs/reference/tooling/TOOL_DISPOSITION.md`, and reconcile the registry's prose row/inventory count with the resulting current table.

```markdown
## Planning capability rebaseline v3 temporary tooling

| Path | Disposition | Owner and rationale |
|---|---|---|
| `tools/lab/planning-capability-rebaseline-v3/scripts/generate_inventory.py` | **TRANSITIONAL** | Planning capability rebaseline v3: read-only frozen-plan inventory/reference generator; migration-local and not planning/evidence authority. |
| `tools/lab/planning-capability-rebaseline-v3/scripts/validate_manifests.py` | **TRANSITIONAL** | Planning capability rebaseline v3: fail-closed source/disposition/lineage/reference validator; migration-local and not production tooling. |
| `tools/lab/planning-capability-rebaseline-v3/scripts/render_operations.py` | **TRANSITIONAL** | Planning capability rebaseline v3: emits a non-mutating JSONL execution plan for ag-planning/ag-ledger; deliberately does not mutate canonical lifecycle state itself. |
```
