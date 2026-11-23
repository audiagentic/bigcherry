# Decode attention accumulator-once (gfx1100): resource gate

Form A retains zero-accumulator gfx11 WMMA per call and a separate f32 add in the original order. It accumulates QK in native layout until softmax and PV until the segment partial output, with the online-softmax rescale still applied elementwise. A compile-time guard selects h256/gqa6/bs16, no window, F16, no packed layout. The active i8-cache flag uses Form A only for chained NS1–4; the fp8-cache flag uses it for chained and unchained NS2–4. The other forms retain the unchanged shim.

Baseline: pinned-source full libr3 build before the decode edit (same toolchain and target). Candidate: niced, serial HIP unit builds from this branch. Resource information is from gfx1100 code-object metadata, not compiler estimates.

The first unguarded prototype increased private-memory spills in six instantiations. Those instantiations were excluded at compile time before this final gate.

**Gate: PASS, pre-link unit comparison.** All **388 decode entries** (194 plain and 194 chained) in twelve attention device units were inspected. Exactly 10 decode entries changed resource metadata; all ten reduced spills and private bytes. No decode entry increased spills or private bytes, and no entry lost theoretical wave32 occupancy. No non-decode kernel in those units changed resource metadata. Baseline and candidate metadata instantiation sets match exactly.

The complete per-instantiation resource table follows; columns show VGPR, spilled VGPR, private bytes and theoretical resident wave32 waves per SIMD. The standalone unit resource check precedes the full plugin link and runtime tests, so it is not a readiness claim.

# Gfx1100 decode resource table (before / Form A)

VGPR / spilled VGPR / private bytes / theoretical wave32 waves per SIMD.
Every entry uses the same pinned-source and toolchain.

| Device unit | Instantiation | Before | Form A |
|---|---|---|---|
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi8ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi8ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 228/0/0/6 | 228/0/0/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi8ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/40/164/6 | 256/40/164/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi8ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/83/336/6 | 256/83/336/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/23/96/6 | 256/23/96/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/45/184/6 | 256/45/184/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 188/0/0/8 | 188/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/32/116/6 | 256/32/116/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/53/208/6 | 256/53/208/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_gqa8 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi8ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi8ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 220/0/0/6 | 220/0/0/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi8ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/36/148/6 | 256/36/148/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi8ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/62/252/6 | 256/62/252/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 169/0/0/9 | 169/0/0/9 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/49/200/6 | 256/49/200/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 190/0/0/8 | 190/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/24/84/6 | 256/24/84/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/52/204/6 | 256/52/204/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 168/0/0/9 | 168/0/0/9 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi8ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/15/64/6 | 256/15/64/6 |
| r4d_attn_paged_gqa8 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi8ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi1ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi1ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 227/0/0/6 | 227/0/0/6 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 186/0/0/8 | 186/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 192/0/0/8 | 192/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 192/0/0/8 | 192/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 170/0/0/9 | 170/0/0/9 |
| r4d_attn_paged_h128_gqa1 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 188/0/0/8 | 188/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi1ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi1ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 218/0/0/7 | 218/0/0/7 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 168/0/0/9 | 168/0/0/9 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 192/0/0/8 | 192/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 190/0/0/8 | 190/0/0/8 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi1ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 167/0/0/9 | 167/0/0/9 |
| r4d_attn_paged_h128_gqa1 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi1ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 190/0/0/8 | 190/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi16ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi16ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 228/0/0/6 | 228/0/0/6 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 192/0/0/8 | 192/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 188/0/0/8 | 188/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi8ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z20r4dchain_attn_decodeILi8ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi16ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi16ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 218/0/0/7 | 218/0/0/7 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 169/0/0/9 | 169/0/0/9 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 168/0/0/9 | 168/0/0/9 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi8ELi16ELi128ELi16ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa16 | `_Z22r4d_attn_decode_kernelILi8ELi16ELi128ELi16ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi2ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi2ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 228/0/0/6 | 228/0/0/6 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 188/0/0/8 | 188/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi2ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi2ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 220/0/0/6 | 220/0/0/6 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 169/0/0/9 | 169/0/0/9 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 190/0/0/8 | 190/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 168/0/0/9 | 168/0/0/9 |
| r4d_attn_paged_h128_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi2ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/16/68/6 | 256/16/68/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 239/0/0/6 | 239/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 246/0/0/6 | 246/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 237/0/0/6 | 237/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 240/0/0/6 | 240/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 236/0/0/6 | 236/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 234/0/0/6 | 234/0/0/6 |
| r4d_attn_paged_h128_gqa2w | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi2ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/16/68/6 | 256/16/68/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/16/68/6 | 256/16/68/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 239/0/0/6 | 239/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 239/0/0/6 | 239/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 246/0/0/6 | 246/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 246/0/0/6 | 246/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 250/0/0/6 | 250/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 237/0/0/6 | 237/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 237/0/0/6 | 237/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 251/0/0/6 | 251/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/1/8/6 | 256/1/8/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 240/0/0/6 | 240/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 240/0/0/6 | 240/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 251/0/0/6 | 251/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 236/0/0/6 | 236/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 236/0/0/6 | 236/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 243/0/0/6 | 243/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 248/0/0/6 | 248/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 256/0/0/6 | 256/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi4ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 234/0/0/6 | 234/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi4ELi16ELi0ELi1580488251ELi1ELj0EEv7R4DArgsi` | 234/0/0/6 | 234/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi4ELi16ELi0ELi20207163ELi1ELj0EEv7R4DArgsi` | 248/0/0/6 | 248/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi4ELi16ELi1ELi20207163ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi4ELi16ELi1ELi422860347ELi1ELj0EEv7R4DArgsi` | 255/0/0/6 | 255/0/0/6 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi7ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi1ELi16ELi128ELi7ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 228/0/0/6 | 228/0/0/6 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi2ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi3ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 188/0/0/8 | 188/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 185/0/0/8 | 185/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z20r4dchain_attn_decodeILi4ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 189/0/0/8 | 189/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi7ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 187/0/0/8 | 187/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi128ELi7ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 220/0/0/6 | 220/0/0/6 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 169/0/0/9 | 169/0/0/9 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 193/0/0/7 | 193/0/0/7 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 190/0/0/8 | 190/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi7ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 168/0/0/9 | 168/0/0/9 |
| r4d_attn_paged_h128_gqa7 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi128ELi7ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 191/0/0/8 | 191/0/0/8 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/73/296/6 | 256/73/296/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj1EEv7R4DArgsi` | 256/58/236/6 | 256/58/236/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj2EEv7R4DArgsi` | 256/52/212/6 | 256/52/212/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj3EEv7R4DArgsi` | 256/73/296/6 | 256/73/296/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj4EEv7R4DArgsi` | 256/50/204/6 | 256/50/204/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/46/188/6 | 256/46/188/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi12ELi4ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/124/500/6 | 256/124/500/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/59/240/6 | 256/59/240/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/39/160/6 | 256/39/160/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/33/136/6 | 256/33/136/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/58/236/6 | 256/58/236/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/39/144/6 | 256/39/144/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/52/212/6 | 256/52/212/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/54/216/6 | 256/54/216/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/42/172/6 | 256/42/172/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/48/180/6 | 256/48/180/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/62/248/6 | 256/62/248/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/52/212/6 | 256/52/212/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/41/168/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/38/156/6 | 256/38/156/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/24/100/6 | 256/24/100/6 |
| r4d_attn_paged_h256_gqa12 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/56/224/6 | 256/56/224/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj1EEv7R4DArgsi` | 256/48/196/6 | 256/48/196/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj2EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj3EEv7R4DArgsi` | 256/55/224/6 | 256/55/224/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi11818555ELi0ELj4EEv7R4DArgsi` | 256/42/172/6 | 256/42/172/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/39/160/6 | 256/39/160/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi12ELi4ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/94/380/6 | 256/94/380/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/22/92/6 | 256/22/92/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/60/244/6 | 256/60/244/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/40/164/6 | 256/40/164/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/27/112/6 | 256/27/112/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/46/188/6 | 256/46/188/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/21/88/6 | 256/21/88/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/208/6 | 256/51/208/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/31/112/6 | 256/31/112/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/208/6 | 256/51/208/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/54/216/6 | 256/54/216/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/44/180/6 | 256/44/180/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/32/116/6 | 256/32/116/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/58/232/6 | 256/58/232/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/18/76/6 | 256/18/76/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/50/204/6 | 256/50/204/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/41/168/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/17/72/6 | 256/17/72/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/27/112/6 | 256/27/112/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_h256_gqa12 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi12ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/190/764/6 | 256/190/764/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/190/764/6 | 256/190/764/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/194/780/6 | 256/194/780/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/231/928/6 | 256/231/928/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/231/928/6 | 256/231/928/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/234/940/6 | 256/234/940/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/184/740/6 | 256/184/740/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/184/740/6 | 256/184/740/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/184/740/6 | 256/184/740/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/205/824/6 | 256/205/824/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/205/824/6 | 256/205/824/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/206/828/6 | 256/206/828/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/189/744/6 | 256/189/744/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/189/744/6 | 256/189/744/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/193/760/6 | 256/193/760/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/330/852/6 | 256/330/852/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/330/852/6 | 256/330/852/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/339/864/6 | 256/339/864/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/180/724/6 | 256/180/724/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/180/724/6 | 256/180/724/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/180/724/6 | 256/180/724/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/184/740/6 | 256/184/740/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/184/740/6 | 256/184/740/6 |
| r4d_attn_paged_h256_gqa2 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/187/752/6 | 256/187/752/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/188/756/6 | 256/188/756/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/188/756/6 | 256/188/756/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/192/772/6 | 256/192/772/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/206/828/6 | 256/206/828/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/206/828/6 | 256/206/828/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/231/928/6 | 256/231/928/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/178/716/6 | 256/178/716/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/178/716/6 | 256/178/716/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/181/728/6 | 256/181/728/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/205/824/6 | 256/205/824/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/205/824/6 | 256/205/824/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/204/820/6 | 256/204/820/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/186/732/6 | 256/186/732/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/186/732/6 | 256/186/732/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/190/748/6 | 256/190/748/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/328/844/6 | 256/328/844/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/328/844/6 | 256/328/844/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/337/856/6 | 256/337/856/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi0ELi1177835067ELi1ELj0EEv7R4DArgsi` | 256/178/716/6 | 256/178/716/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi0ELi1312052795ELi1ELj0EEv7R4DArgsi` | 256/178/716/6 | 256/178/716/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi0ELi1848923707ELi1ELj0EEv7R4DArgsi` | 256/177/712/6 | 256/177/712/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi1ELi137647675ELi1ELj0EEv7R4DArgsi` | 256/182/732/6 | 256/182/732/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi1ELi3429947ELi1ELj0EEv7R4DArgsi` | 256/182/732/6 | 256/182/732/6 |
| r4d_attn_paged_h256_gqa2 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi2ELi16ELi1ELi674518587ELi1ELj0EEv7R4DArgsi` | 256/185/744/6 | 256/185/744/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi4ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/40/164/6 | 256/40/164/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi4ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/87/352/6 | 256/87/352/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/23/96/6 | 256/23/96/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/45/184/6 | 256/45/184/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/32/116/6 | 256/32/116/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/53/208/6 | 256/53/208/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_h256_gqa4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi4ELi16ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/36/148/6 | 256/36/148/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi4ELi16ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/62/252/6 | 256/62/252/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/49/200/6 | 256/49/200/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/24/84/6 | 256/24/84/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/200/6 | 256/51/200/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi4ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/15/64/6 | 256/15/64/6 |
| r4d_attn_paged_h256_gqa4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi4ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/40/164/6 | 256/38/156/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/60/244/6 | 256/60/244/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/49/200/6 | 256/49/200/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/42/172/6 | 256/42/172/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/60/244/6 | 256/60/244/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/83/336/6 | 256/83/336/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/23/96/6 | 256/22/92/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/57/232/6 | 256/37/152/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/48/196/6 | 256/48/196/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/29/120/6 | 256/29/120/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/41/168/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/45/184/6 | 256/45/184/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/32/116/6 | 256/27/96/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/53/216/6 | 256/50/204/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/55/220/6 | 256/55/220/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/45/184/6 | 256/45/184/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/36/132/6 | 256/36/132/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/53/208/6 | 256/53/208/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/18/76/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/40/164/6 | 256/34/132/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/34/140/6 | 256/34/140/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/18/76/6 | 256/18/76/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/34/140/6 | 256/34/140/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/25/104/6 | 256/25/104/6 |
| r4d_attn_paged_h256_gqa6 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/39/160/6 | 256/39/160/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/57/232/6 | 256/57/232/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/47/192/6 | 256/47/192/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/49/200/6 | 256/49/200/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/71/288/6 | 256/71/288/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/59/240/6 | 256/35/144/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/39/160/6 | 256/39/160/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/25/104/6 | 256/25/104/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/32/132/6 | 256/32/132/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/23/96/6 | 256/23/96/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/49/200/6 | 256/49/200/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/24/84/6 | 256/24/84/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/43/176/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/58/232/6 | 256/58/232/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/25/104/6 | 256/25/104/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/33/136/6 | 256/33/136/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/29/104/6 | 256/29/104/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/200/6 | 256/51/200/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/15/64/6 | 256/15/64/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/208/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/37/152/6 | 256/37/152/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/17/72/6 | 256/17/72/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/24/100/6 | 256/24/100/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/16/68/6 | 256/16/68/6 |
| r4d_attn_paged_h256_gqa6 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi16ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/73/296/6 | 256/73/296/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj1EEv7R4DArgsi` | 256/58/236/6 | 256/58/236/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj2EEv7R4DArgsi` | 256/57/232/6 | 256/57/232/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj3EEv7R4DArgsi` | 256/73/296/6 | 256/73/296/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj4EEv7R4DArgsi` | 256/51/208/6 | 256/51/208/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/47/192/6 | 256/47/192/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi1ELi16ELi256ELi6ELi4ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/124/500/6 | 256/124/500/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/59/240/6 | 256/59/240/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/39/160/6 | 256/39/160/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/33/136/6 | 256/33/136/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi2ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/58/236/6 | 256/58/236/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/39/144/6 | 256/39/144/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/52/212/6 | 256/52/212/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/54/216/6 | 256/54/216/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/42/172/6 | 256/42/172/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/48/180/6 | 256/48/180/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi3ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/62/248/6 | 256/62/248/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/26/108/6 | 256/26/108/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/52/212/6 | 256/52/212/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/41/168/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/38/156/6 | 256/38/156/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/24/100/6 | 256/24/100/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z20r4dchain_attn_decodeILi4ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/56/224/6 | 256/56/224/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj1EEv7R4DArgsi` | 256/48/196/6 | 256/48/196/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj2EEv7R4DArgsi` | 256/44/180/6 | 256/44/180/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj3EEv7R4DArgsi` | 256/55/224/6 | 256/55/224/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi11818555ELi0ELj4EEv7R4DArgsi` | 256/43/176/6 | 256/43/176/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi0ELi1186223675ELi0ELj0EEv7R4DArgsi` | 256/40/164/6 | 256/40/164/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi1ELi16ELi256ELi6ELi4ELi1ELi11818555ELi0ELj0EEv7R4DArgsi` | 256/94/380/6 | 256/94/380/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/22/92/6 | 256/22/92/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/60/244/6 | 256/60/244/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/40/164/6 | 256/40/164/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/27/112/6 | 256/27/112/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/46/188/6 | 256/46/188/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/21/88/6 | 256/21/88/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi2ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/208/6 | 256/51/208/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/31/112/6 | 256/31/112/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/51/208/6 | 256/51/208/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/54/216/6 | 256/54/216/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/30/124/6 | 256/30/124/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/44/180/6 | 256/44/180/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/32/116/6 | 256/32/116/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi3ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/58/232/6 | 256/58/232/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi1177835067ELi0ELj0EEv7R4DArgsi` | 256/18/76/6 | 256/18/76/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/50/204/6 | 256/50/204/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj1EEv7R4DArgsi` | 256/41/168/6 | 256/41/168/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj2EEv7R4DArgsi` | 256/17/72/6 | 256/17/72/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj3EEv7R4DArgsi` | 256/27/112/6 | 256/27/112/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi0ELi3429947ELi0ELj4EEv7R4DArgsi` | 256/19/80/6 | 256/19/80/6 |
| r4d_attn_paged_h256_gqa6_bs4 | `_Z22r4d_attn_decode_kernelILi4ELi16ELi256ELi6ELi4ELi1ELi3429947ELi0ELj0EEv7R4DArgsi` | 256/31/128/6 | 256/31/128/6 |

Total decode instantiations: 388
Counts by unit: [('r4d_attn_paged_gqa8', 32, 80), ('r4d_attn_paged_h128_gqa1', 16, 38), ('r4d_attn_paged_h128_gqa16', 20, 42), ('r4d_attn_paged_h128_gqa2', 16, 38), ('r4d_attn_paged_h128_gqa2w', 16, 22), ('r4d_attn_paged_h128_gqa4', 40, 44), ('r4d_attn_paged_h128_gqa7', 16, 38), ('r4d_attn_paged_h256_gqa12', 56, 99), ('r4d_attn_paged_h256_gqa2', 48, 58), ('r4d_attn_paged_h256_gqa4', 16, 44), ('r4d_attn_paged_h256_gqa6', 56, 99), ('r4d_attn_paged_h256_gqa6_bs4', 56, 92)]
Decode resource records changed: 10
Unsafe decode increases: 0
Other resource records changed in measured units: 0

## Full build (before GPU gates)

Full pinned-source gfx1100 `libr3` module and `r3_selftest` executable built under `nice -n 19` and a single compiler job. All 131 Ninja actions completed; the final two linked `radiance_home/kernels/libr3.so` and `r3_selftest`. The plugin generated zero libr4d stubs. Neither byte identity nor attention selftests had run when this build was recorded.


## Stage 3b — Form A bit identity

Validated saved eight context sweeps: **288/288 case PASS; zero differing bytes** across scratch partials and merged output (INT8 and FP8 KV both 144/144). Per KV dtype: ctx 1 = 44, ctx 700 = 36, ctx 4096 = 32, ctx 32768 = 32. Both chain states, sequence counts 1-8, qlen 1-4 (and extended 8/10), ragged inputs and inactive/forced splits were exercised. No cases failed or skipped.

Exact controller: `python3 <scratch>/run-bit-gates.py`; each sweep: `agent-lease acquire gpu --ttl 12m --est 5m --wait -- python3 <scratch>/bit-sweep.py <dtype> <ctx> <baseline-plugin> <candidate-plugin>`. Controller ends `COMPLETE: ALL 8 CONTEXT SWEEPS PASSED (byte exact)`; per-case logs `bit-{i8,fp8}-{1,700,4096,32768}.log` report `diff=0` throughout. **Gate 2 PASS**; selftests and cold replay pending.


## Stage 4 — actual `r3_selftest` cases

Full gfx1100 libr3 plugin and selftest executable were built (131/131 Ninja steps, 115 HIP device units); the test binary was **run** against the candidate plugin in two independently queued short GPU leases. Controller command: `python3 <scratch>/run-selftest-gates.py` (invokes `agent-lease acquire gpu --ttl 12m --est 5m --wait -- <build>/r3_selftest <build>/radiance_home/kernels/libr3.so --case <case>`).

- `attn_decode_h256_gqa6_i8`: **13 OK** (four normal, four numeric, four numeric-negctl, one scratch-coverage).
- `attn_decode_h256_gqa6_bf16kv`: **3 OK** (normal, decode split law over 24 batches, scratch-coverage).

Total **16/16 OK, zero FAIL, zero skip**; two explicit PASS results and controller COMPLETE. Selftest execution **PASS**. Cold replay pending.


## Stage 5 — corrected cold decode replay

Same-process baseline/candidate HIP-event replay, separate queued 12m TTL lease for each of 12 cases; 3 AB/BA/AB repeats per case, at least 512 MiB of rotated KV copies. Full runner: `python3 <scratch>/run-perf-gates.py`; each measured executable `<scratch>/perf[-i8] <baseline-plugin> <candidate-plugin> <ctx> <seqs> <qlen> <chain> <iters> --cold256` executes within `agent-lease acquire gpu --ttl 12m --est 5m --wait -- ...`.

The initial 128-iteration ctx=700/M=1 replay touched only ~92 MiB, so its noisy INT8 median is not an acceptable cold-cache result. Scratch-only controller corrected to `iters=max(128,min(512,ceil(512 MiB / KV-copy bytes)))`, with a minimum of 256 MiB actually touched. A fresh complete 12-case replay with this rule is tabulated below; preliminary logs remain separately saved.

Numbers: median [min-max] microseconds per launch; effective KV throughput in GB/s, computed from the logical cache bytes consumed per launch. The compile-time guard changes chained INT8 flag 1177835067 NS1-4 and FP8 flag 3429947 NS2-4 both chain modes, for h256/gqa6/bs16/f16 only. FP8 NS1 uses the original shim; 378 other decode instantiations retain unchanged code and metadata.

| KV | ctx | seqs x qlen | chained | Baseline us | Form A us | GB/s base / new | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp8 | 32768 | 1x1 | 0 | 296.754 [296.685-371.247] | 292.699 [288.851-329.241] | 113.1 / 114.6 | +1.4% |
| fp8 | 32768 | 6x4 | 0 | 1857.285 [1856.983-1926.906] | 1086.057 [1085.011-1104.892] | 108.4 / 185.4 | +71.0% |
| fp8 | 32768 | 8x4 | 0 | 2442.737 [2440.286-2525.204] | 1418.733 [1417.035-1445.889] | 109.9 / 189.2 | +72.2% |
| fp8 | 700 | 1x1 | 0 | 40.429 [40.309-61.160] | 40.486 [40.396-50.751] | 17.7 / 17.7 | -0.1% |
| fp8 | 700 | 6x4 | 0 | 98.194 [97.884-170.430] | 63.724 [63.639-104.878] | 43.8 / 67.5 | +54.1% |
| fp8 | 700 | 8x4 | 0 | 102.550 [102.123-174.018] | 63.961 [63.469-102.940] | 55.9 / 89.7 | +60.3% |
| i8 | 32768 | 1x1 | 1 | 118.867 [117.896-194.810] | 104.892 [104.630-144.744] | 284.5 / 322.4 | +13.3% |
| i8 | 32768 | 6x4 | 1 | 899.450 [894.939-930.496] | 755.674 [738.003-775.334] | 225.6 / 268.5 | +19.0% |
| i8 | 32768 | 8x4 | 1 | 1173.365 [1171.514-1200.565] | 984.637 [978.013-992.332] | 230.6 / 274.8 | +19.2% |
| i8 | 700 | 1x1 | 1 | 23.989 [23.897-42.435] | 20.249 [20.204-29.936] | 30.1 / 35.7 | +18.5% |
| i8 | 700 | 6x4 | 1 | 46.309 [46.200-123.201] | 37.220 [37.195-82.308] | 93.6 / 116.5 | +24.4% |
| i8 | 700 | 8x4 | 1 | 49.563 [49.518-123.739] | 39.835 [39.624-81.077] | 116.6 / 145.1 | +24.4% |

**Performance gate: PASS.** Ten of ten replayed changed forms improve in median (13.3%-72.2%); the two unchanged FP8 NS1 M=1 paths remain flat within measurement spread. No measured regression outside spread. All 12 cases x 3 repeats completed; no serving, step A/B or KLD.


## Stage 6 — other WMMA sites (candidates only)

Unmodified follow-on candidates: (1) other decode geometry/flag sets in `bc_b16_decode_impl.h`, QK and PV `D::wmma` for `2*(TILE/16)*(HEAD_DIM/16)` calls per KV tile; (2) `bc_b16_prefill_impl.h` F16/BF16 QK (near 496) and PV (near 594), `2*(TILE/16)*(HEAD_DIM/16)` calls per tile, with separate softmax-rescale proof needed; (3) paged-window `r4d_attn_paged_window.hip` BF16 QK near 870 and F16 PV near 959/964, roughly `HEAD_DIM/16` calls each per key tile, with distinct reduction/merge semantics. h128/gqa1,2,4,7,16; h256/gqa2,4,12; gqa8; and h256/gqa6/bs4 remain outside the guard. Six initial spill regressions illustrate why every follow-on variant needs a fresh resource gate. No other kernel was modified. Form B was not built or measured.
