# Backup smoke fixture — pending planning API amendment

Owner direction, 2026-11-27: retain the current tiny Llama CPU fixture and add Qwen3.5-0.8B as a separately qualified backup and GPU smoke fixture. This is nonblocking for the current Radiance 2.3 bump. Existing owners: MEN15 (fixture qualification), MEN24 (bump integration). This evidence is a proposed amendment, not a new authoritative planning record or a completed gate.

## MEN15 amendment

- Preserve the original Llama source and both block-FP8 and IQ8R artifacts, conversion provenance and hashes.
- Acquire original Qwen3.5-0.8B safetensors, config and tokenizer at a pinned revision; verify source hashes. Local GGUF/MQ4 artifacts do not establish native conversion provenance.
- Use the existing converter and store separately named BF16 reference and IQ8R fixtures, recipes, converter/build identities and output hashes.
- Use the existing native YAML fixture mechanism and shared smoke runner with explicit fixture selection and hash verification. Do not automatically retry with a different model when a smoke fails.
- Qualify CPU with all GPUs hidden, and gfx1100/gfx1201 under short GPU leases. Exercise prefill/decode, deterministic nonempty completions, actual backend and selected rows, clean shutdown, IQ8R vs BF16 correctness and quality. Wrong hash, missing fixture and unsupported geometry must fail closed.
- Keep Llama as the fast CPU default until a deliberate migration is warranted and a replacement has passed. Qwen GPU smoke is separate from production-model acceptance.

## MEN24 amendment

Native upstream 2.2 and 2.3 lack h64/gqa1 paged attention, while Qwen3.5-0.8B uses supported h256/gqa4. Existing production text attention uses h256/gqa6 or gqa12; DFlash uses h128/gqa4. Do not widen constraints merely to pass a tiny fixture. Explicitly select a previously qualified compatible fixture during future bumps; record deliberate transitions and exact model/engine/plugin identities. Do not delay current production-model tests for this follow-up.

## Planning access failure

Combined update rejected: `Command too long (max 2000 chars)`. The following individual API update (host paths normalized to runbook variables) was subsequently rejected with `Command blocked for security`:

```sh
uv run --project $AUDIAGENTIC_ROOT --no-sync python -c 'from pathlib import Path; from audiagentic.components.planning import planning_api as a; print(a.update_item(Path("$BC_WORKTREE"),"MEN15",{"notes":"Owner 2026-11-27: preserve both Llama fixtures. Add Qwen3.5-0.8B as named CPU fallback and GPU smoke. Pin original checkpoint and native BF16/IQ8R conversion recipes and hashes. Qualify CPU, gfx1100 and gfx1201 with existing native YAML runner, deterministic completions, row selection, clean shutdown and BF16 comparison. Explicit selection only; never silently switch after failure. Nonblocking for this bump."}),flush=True)'
```

The connected tool list does not expose ag-planning. Per docs/reference/agents/PLANNING_ACCESS.md, do not edit authoritative Markdown/YAML manually. Apply these amendments via the approved planning API once access is restored; verify and commit the resulting records. Neither MEN15 nor MEN24 has been updated with this amendment yet.
