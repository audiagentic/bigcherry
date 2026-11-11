# rad0016_vtp30_native_tp3_ownership

**Status:** untested
**Plan item:** VTP13

## Summary

Experimental native Radiance Qwen3.8-27B original BF16 **three-rank unequal weight declarations**. The existing architecture plugin, builder, weight loader and GDN/attention/FFN model graph remain the only implementation. Uses already-supported per-weight spans rather than importing VTP's separate HIP programs.

## Scope

Enabled by the **separate experimental source recipe** `radiance-vtp30-native-tp3` and explicit `BC_VTP_NATIVE_TP3=1` on the exact 64L/5120/24Q/4KV/256D/17408FF/248320-vocab BF16 model with world size 3. For all other inputs the existing geometry path remains unchanged or the opt-in is explicitly refused. Four physical rank head/FFN dimensions and stacked GDN source spans are source-checked; Radiance's existing exact world-size-three collectives, native model residuals and recurrent/KV caches remain untouched.

## Evidence and limits

The eight managed source edits apply atomically to five pinned v1.17.0 files, remain idempotent and refuse source drift. The patched native Qwen BF16 plugin passes GCC C++23 syntax compilation with original v1.17.0 headers; ten targeted VTP patch tests pass. The `untested` state is intentional: no native mixed-architecture inference or end-to-end quality run has occurred.

**Hard blocker:** Radiance's engine currently discards GPUs of a different architecture from the first, and its kernel loader checks code eligibility against the first device only. These safety checks are **not** disabled here. Mixed gfx1100/gfx1201 startup requires per-rank code selection and all-operator kernel verification before GPU testing.

No production source, default, ABI, scheduler, state allocator, host collective or sampling code is changed. Native throughput and 262K context remain unvalidated.
