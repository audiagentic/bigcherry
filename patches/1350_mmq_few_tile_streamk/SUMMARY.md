# 1350_mmq_few_tile_streamk

**Status:** validated  
**Plan item:** QFP37

## What it does

Compiles a second specialization of llama.cpp b11474's existing MMQ Stream-K kernel/fixup path. With `BIGCHERRY_MMQ_FEW_TILE_STREAMK=1`, ordinary Q8_0 fallback MMQ on exact gfx1100/gfx1201 can select that specialization when the physical shape is underfilled: at most eight output-row tiles, fewer destination tiles than device CUs, at least eight K tiles, at least 128 activation columns, and bounded fixup storage.

The package default stays off. The Flash-Next runtime profile enables it because that is the qualified model/topology.

## Promotion

QFP18 lightweight promotion tier, Brutus 2026-10-08, pin b11474. `b-metamem-qfp37c` composed 49/49, compiled cleanly, smoke had 0 error lines, and `PATCH_HIT` was present.

Prefill ABBA gain was +1.8% / +1.2% / +1.9% at 8K / 24K / 98K with complete separation at every depth. Decode did not regress. The Stream-K arm is not bit-identical by design; 24 CPU-F32-reference probes were no worse than production (top-1 24/24 vs 23/24, TV mean 0.0738 vs 0.0764). The off arm is the upstream MMQ dispatch.

## Scope

- ordinary MMQ only: no `ids_dst`/MoE;
- Q8_0 fallback specialization only;
- exact gfx1100/gfx1201;
- physical-shape/occupancy gate, not exact model dimensions;
- existing I/J/thread/SRAM/K geometry and upstream Stream-K fixup retained.

Not measured: second model. Build/runtime default remains off outside the qualified Flash-Next profile.
