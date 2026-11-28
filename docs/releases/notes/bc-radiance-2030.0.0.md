# bc-radiance-2030.0.0

BigCherry on Radiance `3b13fd32663ef52252a7f0be255523480aa22ffe` (upstream `3b13fd32663e`).

## Radiance 2.3.0

BigCherry now builds on upstream Radiance v2.3.0 (previously 2.2.0).

**What changed**
- All 44 managed patches ported to 2.3.0, including upstream's renamed quantised block headers; plugin ABI 45 combines upstream ABI 44 with the retained local layout extensions, and every component is rebuilt against it.
- Upstream 2.3.0 brings per-card memory headroom, queue reserve, failed-handoff memory ownership, scheduler admission-error handling and loader fixes; they are used as upstream ships them.
- The bc-iqr overlay merged upstream's 2.3.0 changes to the IQR embedding, vocabulary head, IQR GEMM and int8 prefill units; the compiled IQR code is unchanged.

**Measured**
- Qwen 27B IQ8R on three cards, eight streams, against the 2.2 release 2020.24.1 in the same session (24 trials, 3 seeds): no regression; paired medians +0.8% and -1.4%, decode step time level.
- All production layouts (Qwen 27B two and three cards, IQ6R-V4 three cards, Flash-Next four cards and two XTX plus R9700) start, serve and stop cleanly.

**Checks**
- pin-bump: drift, audit, patch lint, coverage, apply, compile, lane build, CPU smoke and re-audit pass; component tests and the offline suite pass.

**Known limitations**
- Short-prompt eight-stream throughput on three cards remains about 5% below the 2020.8.5 high point; that gap predates 2.3.0 and is tracked separately.
