# 1003_quantized_cpy_thread_block_fix

## Summary

Was intended to fix eleven quantized f32<->{q8_0,q4_0,q4_1,q5_0,q5_1}/f32->iq4_nl copy-kernel launches in cpy.cu that computed one block per quantized element/block and launched with a hardcoded 1 thread per block, instead of using CUDA_CPY_BLOCK_SIZE like the rest of the file.

## Outcome

Recorded state: `rejected` ([patch.toml](patch.toml)); qualification and limitations: [SUMMARY.md](SUMMARY.md).

## Detail

The patch package's [SUMMARY.md](SUMMARY.md) records the rationale, scope, and available evidence. See [patch.toml](patch.toml) for the authoritative lifecycle state and dependencies.
