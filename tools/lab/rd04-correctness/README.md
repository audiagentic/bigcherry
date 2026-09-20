# rd04-correctness: ad-hoc real-hardware driver for RD04 contract correctness

Plan item: RD04 / PA36 (1202 migration)
Status: active
Owner: PA36 (patching-patch-system)
Question state: open

## Question

On real hardware, does `run_rd04_contract_correctness()` pass for
`RD04-BF16-FLASH-ATTN-TILE` across its declared architectures
(`gfx1100`/`gfx1201`/`gfx1030`)? No `--run-rd04-correctness` CLI flag
exists yet, so this driver loops all three architectures, setting the matching
device index per iteration (never both `HIP_VISIBLE_DEVICES` and
`ROCR_VISIBLE_DEVICES` to the same index — that double-filters to zero
devices, a known trap in this project).

## Inputs

- `/home/audumla/rocm-shim`; device map `gfx1100=0`, `gfx1201=2`,
  `gfx1030=3`; `tierA-qwen4b-q6k` model; a small wikitext-2 slice corpus.

## Outputs

Per-architecture RD04 correctness results written to the campaign workdir.

## Runtime

GPU required: yes (all three architectures)
Real compilation required: yes (control + subject builds)
Mutates canonical BigCherry state: no

## Safety

- Direct call to the real `validation_campaign` producer function; not
  production tooling and not the evidence authority.
- Check for live build/lease state before running on a shared tree.

## Disposition

Retained (TRANSITIONAL) as the RD04 real-hardware driver; diagnostic
hardware evidence, not a maintained tool.
