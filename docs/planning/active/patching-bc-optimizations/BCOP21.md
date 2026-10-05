---
id: BCOP21
order: 21
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-05T04:50:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Root-cause HIP D=72 Flash Attention fault before retaining fallback

## Description

Backfill of the earlier PHA08 audit. Upstream #28664 closed without merge while #28608 retained evidence of gfx1100 faults in `flash_attn_tile<72,72,64,1,false>`. The fallback should remain narrow while D=72 tail/storage invariants are investigated on the current pin.

## Steps

1. Reproduce on current upstream/BigCherry pin with D=64/72/80 controls on gfx1100 and gfx1201.
2. Inspect vector/tail bounds, padded sequence extents, shared-memory offsets, workspace sizing, generated ISA and resource use.
3. Test the D=72 tail/storage-invariant hypothesis rather than assuming tile width is causal.
4. Retain only HIP+CLIP+d_head==72 fallback while root cause is unresolved; no generic HIP FA disable.
5. If a kernel fix is found, run >=20 repeated 2048/2560px encodes plus parity against matmul attention.

## Related

PHA08; llama.cpp #28608/#28664.

## Acceptance Criteria

- Current-pin reproduction and D=64/72/80 controls exist on AMD targets.
- Root cause is proven or fallback is explicitly retained with rationale.
- Kernel fix promotes only with >=5% VLM encode gain over fallback and no healthy-shape regression >2%.
