# rad0061 — native IQ8R scale precision (RUM15)

**Status:** diagnostic
**Plan item:** RUM15

No production recipe selects this patch.

## Summary

Preserve the exact FP32 product of the U8 per-group and BF16 per-row IQ8R
scales in the existing native `libr4d` group-256 relayout and GEMM readers;
use BigCherry's existing native rewrite of that source for the XTX `libr3`
plugin. The CPU `libref` reference is unchanged.

## Outcome

The fully rebuilt R9700 native IQ8R selftest passed 35/35 checks without
loosening its numeric threshold. The XTX/XTX/R9700 checkpoint-backed C06
attention graph passes all four cases in both GPU rank orders; 108 native
instances are retired per order. The two orders match exactly, and the
GPU's 36,864 QKV BF16 elements all match CPU `libref`. One of 12,288 final
BF16 elements differs from the CPU oracle. The 14 anchored source edits
are idempotent; 23/23 focused Python tests pass.

## Detail

Unselected diagnostic patch, not production TP3. Source scale layout and
cache fingerprint are changed; additional VRAM per resident scale group,
performance, reload/refill admission, 16/64-row attention and actual TP1
activation parity remain to be validated. Evidence:
`docs/evidence/2026-11-22-rum15-f32-native-candidate/README.md`.
