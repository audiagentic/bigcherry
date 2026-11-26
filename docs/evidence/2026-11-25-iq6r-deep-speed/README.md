# IQ6R deep-speed candidate A: embedding inverse-Hadamard exchange

## Decision

Keep the gfx1100-only, compile-time guarded wave-exchange path for rotated IQR embedding forms. The fallback on other targets and all unrotated forms remains the original seven-stage shared-memory transform.

The 128-thread block computes five butterfly stages wholly within each wave using lane shuffles and keeps the 32- and 64-element stages in shared memory. The reduction is from 15 to 5 workgroup barriers per rotated CTA. The low-lane minus high-lane operand ordering is preserved.

## Compilation and resource gate

Both native baseline and experimental variants compiled and linked with the embedding compile unit selected, and the final guarded rewrite separately compiled and linked. The gfx1100 code objects contain the same 17 embedding specialisations. All 17 resource records in the final guarded build are identical to the experimental build.

The attached `candidate-a-resources.csv` lists each form and baseline/candidate registers, spills, private segment, LDS and block size. VGPR spills and SGPR spills are zero on all forms; there are no private-segment or LDS increases. The rotated table B4 form increased from seven to eight VGPRs, below the hardware wave-occupancy cap in both cases, so no resident wave is lost. Other 16 forms retain the same compiler resources.

## GPU bit-identity gate

The baseline and wave-exchange libraries each passed the isolated `r3_selftest --case embed_lookup_iqr` on gfx1100 with no failures: **17 forms × two multipliers (1.0 and 3.25) = 34 exact BF16 output comparisons per library**. Each case compares with the host widening oracle and includes invalid IDs, the final valid row, and the 255/256 trellis run boundary. The test loader was temporarily restricted to the selected embedding test because the one-unit build intentionally omits unrelated diagnostic exports; no test-loader modification is part of this candidate.

The final guarded-code build was also run through the same 17-form GPU byte gate: 34/34 exact comparisons, zero failures, exit status zero.

## GPU replay gate

A one-token M=1, n_embd=5120, n_vocab=512 embedding replay ran in two opposite library orders under separate queued leases. Each form used 12 cold GPU-event samples (after a 256 MiB cache-flush allocation was overwritten) and 40 warm GPU-event samples. The full per-form results and both execution orders are in `candidate-a-replay.csv`.

For the **10 transformed rotated forms**, the geometric mean of the two order-specific candidate/baseline ratios gives a median of **0.98052 cold** and **0.97859 warm**. Every transformed form improved in both cold and warm paired-order aggregates (cold ratios 0.97686–0.98333; warm 0.96767–0.99640). The other seven forms compile to the unchanged original path and serve as noise controls; their small apparent mixed timing changes, including increases, are measurement variation rather than changed kernels.

## Limits

The tests are kernel-unit and embedding selftests, not a serving run, A/B test, or end-to-end step-time measurement. The plugin was built with the embedding unit selected, not as an unrestricted all-kernel production plugin. No serving-level speedup is claimed. This candidate does not modify any non-embedding kernel or target outside the guarded architecture.
