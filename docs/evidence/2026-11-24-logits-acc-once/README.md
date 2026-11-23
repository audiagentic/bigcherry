# Logits WMMA Form A — gfx1100

## Scope

Only the MT=2, FKB=2, I8=true, MH=1, TB=0 WMMA accumulator (chained and standalone) uses the native-layout Form A. Other LM instantiations and M<=2 matvec retain the stock shim. No vendor or gfx1201 changes; Form B was not built.

## Build and linked-unit resource gate (PASS)

- Full plugin and r3_selftest build: `nice -n 19 cmake --build <build-dir> --target libr3 r3_selftest -j 1`, exit 0, final 44/44 steps (incremental after prior failed/aborted iterations recorded in the external stage report).
- Compare 2020.8.1 stock and prechange full libr3 device ELF (identical on all 140 LM records) with the logits device ELF extracted from the **linked** candidate libr3.so .hip_fatbin; inspect metadata using `llvm-readobj --notes`. No GPU was used for resource inspection.
- The 140 standalone and chained instantiations and 42 other kernels in the unit match by identity. No new spills, private-memory increases or waves/SIMD losses. Target unchained 240/12/52 -> 230/0/0 VGPR/spill/private bytes; chained 240/15/64 -> 232/0/0.

# gfx1100 logits WMMA resources: VGPR/spill/private bytes/wave32 waves per SIMD

| Entry | MT | FKB | I8 | MH | TB | before | Form A |
|---|---:|---:|---:|---:|---:|---|---|
| lm_wmma_kernel | 1 | 1 | 0 | 1 | 0 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 0 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 2 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 3 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 4 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 5 | 118/0/0/13 | 118/0/0/13 |
| lm_wmma_kernel | 1 | 1 | 1 | 1 | 6 | 117/0/0/13 | 117/0/0/13 |
| lm_wmma_kernel | 1 | 2 | 0 | 1 | 0 | 148/0/0/10 | 148/0/0/10 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 0 | 179/0/0/8 | 179/0/0/8 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 2 | 223/0/0/6 | 223/0/0/6 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 3 | 215/0/0/7 | 215/0/0/7 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 4 | 180/0/0/8 | 180/0/0/8 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 5 | 197/0/0/7 | 197/0/0/7 |
| lm_wmma_kernel | 1 | 2 | 1 | 1 | 6 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 2 | 1 | 0 | 1 | 0 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 0 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 2 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 3 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 4 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 5 | 167/2/12/9 | 167/2/12/9 |
| lm_wmma_kernel | 2 | 1 | 1 | 1 | 6 | 168/0/0/9 | 168/0/0/9 |
| lm_wmma_kernel | 2 | 2 | 0 | 1 | 0 | 194/0/0/7 | 194/0/0/7 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 0 | 240/12/52/6 | 230/0/0/6 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 2 | 240/12/52/6 | 240/12/52/6 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 3 | 240/12/52/6 | 240/12/52/6 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 4 | 240/12/52/6 | 240/12/52/6 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 5 | 240/14/60/6 | 240/14/60/6 |
| lm_wmma_kernel | 2 | 2 | 1 | 1 | 6 | 240/12/52/6 | 240/12/52/6 |
| lm_wmma_kernel | 3 | 1 | 0 | 1 | 0 | 216/2/12/7 | 216/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 0 | 215/2/12/7 | 215/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 2 | 215/2/12/7 | 215/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 3 | 215/2/12/7 | 215/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 4 | 215/2/12/7 | 215/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 5 | 216/2/12/7 | 216/2/12/7 |
| lm_wmma_kernel | 3 | 1 | 1 | 1 | 6 | 216/2/12/7 | 216/2/12/7 |
| lm_wmma_kernel | 3 | 2 | 0 | 1 | 0 | 234/0/0/6 | 234/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 0 | 234/0/0/6 | 234/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 2 | 234/0/0/6 | 234/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 3 | 234/0/0/6 | 234/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 4 | 234/0/0/6 | 234/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 5 | 235/0/0/6 | 235/0/0/6 |
| lm_wmma_kernel | 3 | 2 | 1 | 1 | 6 | 235/0/0/6 | 235/0/0/6 |
| lm_wmma_kernel | 4 | 1 | 0 | 1 | 0 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 0 | 2 | 0 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 0 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 2 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 3 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 4 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 5 | 239/0/0/6 | 239/0/0/6 |
| lm_wmma_kernel | 4 | 1 | 1 | 1 | 6 | 219/0/0/7 | 219/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 0 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 2 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 3 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 4 | 210/0/0/7 | 210/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 5 | 218/0/0/7 | 218/0/0/7 |
| lm_wmma_kernel | 4 | 1 | 1 | 2 | 6 | 211/0/0/7 | 211/0/0/7 |
| lm_wmma_kernel | 4 | 2 | 0 | 1 | 0 | 240/10/44/6 | 240/10/44/6 |
| lm_wmma_kernel | 4 | 2 | 0 | 2 | 0 | 240/18/76/6 | 240/18/76/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 0 | 240/14/60/6 | 240/14/60/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 2 | 240/14/60/6 | 240/14/60/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 3 | 240/14/60/6 | 240/14/60/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 4 | 240/14/60/6 | 240/14/60/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 5 | 239/16/68/6 | 239/16/68/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 1 | 6 | 239/12/52/6 | 239/12/52/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 0 | 240/22/92/6 | 240/22/92/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 2 | 240/22/92/6 | 240/22/92/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 3 | 240/22/92/6 | 240/22/92/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 4 | 240/22/92/6 | 240/22/92/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 5 | 239/24/100/6 | 239/24/100/6 |
| lm_wmma_kernel | 4 | 2 | 1 | 2 | 6 | 239/20/84/6 | 239/20/84/6 |
| r4dchain_lm_wmma | 1 | 1 | 0 | 1 | 0 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 0 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 2 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 3 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 4 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 5 | 120/0/0/12 | 120/0/0/12 |
| r4dchain_lm_wmma | 1 | 1 | 1 | 1 | 6 | 119/0/0/12 | 119/0/0/12 |
| r4dchain_lm_wmma | 1 | 2 | 0 | 1 | 0 | 150/0/0/10 | 150/0/0/10 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 0 | 181/0/0/8 | 181/0/0/8 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 2 | 225/0/0/6 | 225/0/0/6 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 3 | 217/0/0/7 | 217/0/0/7 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 4 | 182/0/0/8 | 182/0/0/8 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 5 | 199/0/0/7 | 199/0/0/7 |
| r4dchain_lm_wmma | 1 | 2 | 1 | 1 | 6 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 2 | 1 | 0 | 1 | 0 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 0 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 2 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 3 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 4 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 5 | 168/3/16/9 | 168/3/16/9 |
| r4dchain_lm_wmma | 2 | 1 | 1 | 1 | 6 | 168/2/12/9 | 168/2/12/9 |
| r4dchain_lm_wmma | 2 | 2 | 0 | 1 | 0 | 196/0/0/7 | 196/0/0/7 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 0 | 240/15/64/6 | 232/0/0/6 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 2 | 240/15/64/6 | 240/15/64/6 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 3 | 240/15/64/6 | 240/15/64/6 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 4 | 240/15/64/6 | 240/15/64/6 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 5 | 240/0/0/6 | 240/0/0/6 |
| r4dchain_lm_wmma | 2 | 2 | 1 | 1 | 6 | 240/15/64/6 | 240/15/64/6 |
| r4dchain_lm_wmma | 3 | 1 | 0 | 1 | 0 | 214/0/0/7 | 214/0/0/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 0 | 216/3/16/7 | 216/3/16/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 2 | 216/3/16/7 | 216/3/16/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 3 | 216/3/16/7 | 216/3/16/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 4 | 216/3/16/7 | 216/3/16/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 5 | 216/4/20/7 | 216/4/20/7 |
| r4dchain_lm_wmma | 3 | 1 | 1 | 1 | 6 | 216/4/20/7 | 216/4/20/7 |
| r4dchain_lm_wmma | 3 | 2 | 0 | 1 | 0 | 236/0/0/6 | 236/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 0 | 236/0/0/6 | 236/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 2 | 236/0/0/6 | 236/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 3 | 236/0/0/6 | 236/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 4 | 236/0/0/6 | 236/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 5 | 237/0/0/6 | 237/0/0/6 |
| r4dchain_lm_wmma | 3 | 2 | 1 | 1 | 6 | 237/0/0/6 | 237/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 0 | 1 | 0 | 220/0/0/6 | 220/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 0 | 2 | 0 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 0 | 220/0/0/6 | 220/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 2 | 220/0/0/6 | 220/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 3 | 220/0/0/6 | 220/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 4 | 220/0/0/6 | 220/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 5 | 240/2/12/6 | 240/2/12/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 1 | 6 | 221/0/0/6 | 221/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 0 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 2 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 3 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 4 | 212/0/0/7 | 212/0/0/7 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 5 | 221/0/0/6 | 221/0/0/6 |
| r4dchain_lm_wmma | 4 | 1 | 1 | 2 | 6 | 213/0/0/7 | 213/0/0/7 |
| r4dchain_lm_wmma | 4 | 2 | 0 | 1 | 0 | 240/12/52/6 | 240/12/52/6 |
| r4dchain_lm_wmma | 4 | 2 | 0 | 2 | 0 | 240/20/84/6 | 240/20/84/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 0 | 240/16/68/6 | 240/16/68/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 2 | 240/16/68/6 | 240/16/68/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 3 | 240/16/68/6 | 240/16/68/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 4 | 240/16/68/6 | 240/16/68/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 5 | 240/14/60/6 | 240/14/60/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 1 | 6 | 240/13/56/6 | 240/13/56/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 0 | 240/24/100/6 | 240/24/100/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 2 | 240/24/100/6 | 240/24/100/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 3 | 240/24/100/6 | 240/24/100/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 4 | 240/24/100/6 | 240/24/100/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 5 | 240/22/92/6 | 240/22/92/6 |
| r4dchain_lm_wmma | 4 | 2 | 1 | 2 | 6 | 240/21/88/6 | 240/21/88/6 |

code-objects baseline=libr3.so.66.hipv4-amdgcn-amd-amdhsa--gfx1100, candidate=plugin-logits-linked-gfx1100.o
LM variants=140, baseline total=182, candidate total=182. 2020.8.1 stock against pre-change full-plugin LM resource differences: 0.
Non-LM records: before=42 after=42 changed=0 missing=0 added=0
New LM VGPR spills=[], private increases=[], waves losses=[]

## Form A byte-identity gate (PASS)

- Used only short, queued GPU measurement leases, released after each harness invocation; no serving, ms/step A/B, or KLD. Each case compares every f32 output byte between the prechange and new full plugin on deterministic BF16 input, packed weights and scales for M=1..64.
- Main head: N×K=248320×5120, 4097×512, 4163×128 and 82773×5120; each M=1..64 on int8 unchained, int8 chained and E4M3 exported unchained paths. 12 cases ×64 = **768 PASS, zero differences**. A separate N×K=1280×512 smoke reran all three paths ×64 = 192 PASS.
- IQR shared logits WMMA body: N×K=1008×640 and 4097×512, table bits TB=2..6, chained and unchained; 20 cases ×64 = **1280 PASS, zero differences**.
- Total main and IQR checked: **2048 M-values**, zero full-output f32 byte differences, no skipped test of an exposed row, no device error.
- E4M3 has no registered logits_gemm_fp8 plugin row; its exported r4d_logits_gemm_fp8 function was used for unchained byte checking. E4M3 chained dispatch is not exported and was not claimed tested. Only int8 MT2/FKB2 is modified by Form A, and both its chained and unchained paths were tested.

## Full-plugin r3_selftest logits cases (PASS)

Ran the actual linked `r3_selftest` binary against the freshly built `libr3.so`, one queued GPU lease per case, `--case` selectors, zero failures and zero skips:

| Case | OK assertions | Fail | Skip |
|---|---:|---:|---:|
| logits_gemm_i8 | 28 | 0 | 0 |
| logits_gemm_iqr | 88 | 0 | 0 |
| logits_gemm_bf16_cols | 21 | 0 | 0 |
| logits_gemm_vocab_iqr | 67 | 0 | 0 |
| logit_rerank_i8 | 28 | 0 | 0 |
| logit_rerank_iqr | 88 | 0 | 0 |
| **Total** | **320** | **0** | **0** |

Full-plugin build and selftest execution both actually completed; no skipped test case.

## Cold replay measurement gate (PASS)

- Run `replay <stock-libr3.so> <candidate-libr3.so> <N> 5120 i8` under a short queued GPU lease for N=248320 and N=82773. Each row uses 256 MiB D2D cache flush per launch, three rotating weight copies, three repeats, 20 cold kernel launches per repeat, and interleaved stock/candidate arms. Full 228 records preserved in the external scratch run.
- All M17–32 comparisons improve beyond separated repeat ranges; M1/2, M3–16, and M33–64 show no spread-separated regression. Main M32 median 2996.24 -> 2315.39 microseconds (-22.72%), shard M32 965.97 -> 763.97 (-20.91%).

# Cold replay: 256 MiB D2D flush; three rotated weight copies; three repeats

Timing values are kernel-event medians (microseconds) over 20 cold launches in each repeat. Spread is min..max across the three repeats.

| N×K | dtype | M | stock median µs (spread) | Form A median µs (spread) | delta | spread separation |
|---|---|---:|---:|---:|---:|---|
| 82773×5120 | logits_gemm_i8 | 1 | 511.37 (510.89–511.41) | 511.25 (511.21–511.49) | -0.02% | overlap |
| 82773×5120 | logits_gemm_i8 | 2 | 515.37 (514.97–515.69) | 515.13 (514.93–515.41) | -0.05% | overlap |
| 82773×5120 | logits_gemm_i8 | 4 | 631.13 (625.09–631.25) | 631.45 (629.01–632.13) | +0.05% | overlap |
| 82773×5120 | logits_gemm_i8 | 8 | 638.97 (638.65–640.33) | 640.37 (639.61–641.25) | +0.22% | overlap |
| 82773×5120 | logits_gemm_i8 | 16 | 652.41 (651.85–654.01) | 650.81 (650.77–652.49) | -0.25% | overlap |
| 82773×5120 | logits_gemm_i8 | 17 | 934.21 (933.73–936.21) | 737.09 (736.77–737.17) | -21.10% | faster |
| 82773×5120 | logits_gemm_i8 | 20 | 947.97 (947.17–951.25) | 745.89 (743.57–749.21) | -21.32% | faster |
| 82773×5120 | logits_gemm_i8 | 21 | 952.33 (947.73–953.21) | 746.81 (745.57–748.57) | -21.58% | faster |
| 82773×5120 | logits_gemm_i8 | 24 | 959.65 (954.85–959.89) | 752.05 (750.01–752.49) | -21.63% | faster |
| 82773×5120 | logits_gemm_i8 | 25 | 961.25 (960.17–964.41) | 754.25 (754.13–757.29) | -21.53% | faster |
| 82773×5120 | logits_gemm_i8 | 28 | 966.41 (965.85–966.73) | 759.81 (755.93–762.93) | -21.38% | faster |
| 82773×5120 | logits_gemm_i8 | 29 | 967.73 (967.05–969.21) | 761.41 (757.17–762.53) | -21.32% | faster |
| 82773×5120 | logits_gemm_i8 | 32 | 965.97 (965.29–977.05) | 763.97 (760.13–765.13) | -20.91% | faster |
| 82773×5120 | logits_gemm_i8 | 33 | 1339.02 (1336.02–1344.54) | 1343.10 (1340.82–1350.10) | +0.30% | overlap |
| 82773×5120 | logits_gemm_i8 | 37 | 1366.38 (1351.38–1377.78) | 1365.82 (1362.34–1373.22) | -0.04% | overlap |
| 82773×5120 | logits_gemm_i8 | 41 | 1380.82 (1361.54–1382.42) | 1382.54 (1370.50–1387.26) | +0.12% | overlap |
| 82773×5120 | logits_gemm_i8 | 45 | 1377.42 (1375.66–1398.02) | 1373.90 (1371.10–1375.66) | -0.26% | overlap |
| 82773×5120 | logits_gemm_i8 | 48 | 1394.46 (1393.38–1395.18) | 1392.58 (1392.22–1399.78) | -0.13% | overlap |
| 82773×5120 | logits_gemm_i8 | 64 | 2052.43 (2042.27–2060.35) | 2041.79 (2037.31–2061.87) | -0.52% | overlap |
| 248320×5120 | logits_gemm_i8 | 1 | 1501.98 (1501.30–1505.50) | 1502.90 (1501.54–1503.50) | +0.06% | overlap |
| 248320×5120 | logits_gemm_i8 | 2 | 1509.14 (1508.18–1511.22) | 1509.90 (1507.98–1510.18) | +0.05% | overlap |
| 248320×5120 | logits_gemm_i8 | 4 | 1908.18 (1892.42–1910.74) | 1911.46 (1895.14–1920.18) | +0.17% | overlap |
| 248320×5120 | logits_gemm_i8 | 8 | 1927.78 (1917.94–1928.30) | 1924.42 (1920.74–1927.50) | -0.17% | overlap |
| 248320×5120 | logits_gemm_i8 | 16 | 1953.38 (1949.54–1960.54) | 1950.78 (1945.78–1984.94) | -0.13% | overlap |
| 248320×5120 | logits_gemm_i8 | 17 | 2866.51 (2865.55–2876.99) | 2237.91 (2234.03–2242.63) | -21.93% | faster |
| 248320×5120 | logits_gemm_i8 | 20 | 2890.59 (2880.83–2913.71) | 2253.83 (2251.59–2262.87) | -22.03% | faster |
| 248320×5120 | logits_gemm_i8 | 21 | 2910.95 (2900.47–2931.59) | 2266.07 (2259.99–2269.79) | -22.15% | faster |
| 248320×5120 | logits_gemm_i8 | 24 | 2949.47 (2917.07–2961.99) | 2302.63 (2285.43–2306.07) | -21.93% | faster |
| 248320×5120 | logits_gemm_i8 | 25 | 2944.76 (2927.11–2980.95) | 2286.43 (2275.27–2310.23) | -22.36% | faster |
| 248320×5120 | logits_gemm_i8 | 28 | 2968.80 (2928.11–2986.23) | 2290.63 (2286.43–2317.95) | -22.84% | faster |
| 248320×5120 | logits_gemm_i8 | 29 | 2996.07 (2981.91–2996.11) | 2308.31 (2301.83–2326.63) | -22.96% | faster |
| 248320×5120 | logits_gemm_i8 | 32 | 2996.24 (2971.36–3003.27) | 2315.39 (2309.51–2320.67) | -22.72% | faster |
| 248320×5120 | logits_gemm_i8 | 33 | 4027.05 (4022.57–4027.73) | 4022.61 (4020.81–4034.17) | -0.11% | overlap |
| 248320×5120 | logits_gemm_i8 | 37 | 4069.45 (4064.05–4079.37) | 4100.49 (4060.77–4101.25) | +0.76% | overlap |
| 248320×5120 | logits_gemm_i8 | 41 | 4113.13 (4107.89–4121.37) | 4111.77 (4111.01–4117.61) | -0.03% | overlap |
| 248320×5120 | logits_gemm_i8 | 45 | 4154.01 (4146.29–4176.65) | 4171.69 (4151.69–4175.01) | +0.43% | overlap |
| 248320×5120 | logits_gemm_i8 | 48 | 4176.81 (4170.53–4179.57) | 4181.81 (4173.21–4181.93) | +0.12% | overlap |
| 248320×5120 | logits_gemm_i8 | 64 | 6121.67 (6093.71–6122.19) | 6112.91 (6104.07–6123.51) | -0.14% | overlap |

## Effective head-weights bandwidth
- 82773×5120, logits_gemm_i8, M=16: 430419600 weight bytes; stock=659.74 GB/s, Form A=661.36 GB/s
- 82773×5120, logits_gemm_i8, M=32: 430419600 weight bytes; stock=445.58 GB/s, Form A=563.40 GB/s
- 248320×5120, logits_gemm_i8, M=16: 1291264000 weight bytes; stock=661.04 GB/s, Form A=661.92 GB/s
- 248320×5120, logits_gemm_i8, M=32: 1291264000 weight bytes; stock=430.96 GB/s, Form A=557.69 GB/s

Analysed 228 timed repeat records, 38 arm comparisons.

Cold gate PASS: 17–32 gains beyond repeat spread, and no measured regressions in other groups.

## Final source review and remaining candidates

- The only BF16 f32 gfx12 WMMA shim in the logits source is in `lm_wmma_body`: one call per K16 iteration for each token tile, across eight K16 iterations in one BF16 scale block. The new native-layout accumulation is compile-time gated to MT=2, FKB=2, I8=true, MH=1, TB=0. Other LM geometry/format/table variants preserve the original shim, avoiding the 54 spill regressions found with the unrestricted rewrite.
- The M<=2 `mvm_kernel` is not part of the WMMA body and remains unchanged. Adjacent `r4d_gemm_fp8a8.hip` uses an integer I32 WMMA rather than this BF16 f32 shim; it is not a candidate for this same transformation.
- Future independent gfx1100 BF16 shim audits include `r4d_lora_bf16.hip`, `r4d_moe.hip`, `r4d_gdn_wmma.h`, `r4d_dt16.h`, and `r4d_attn_paged_window.hip`. They were identified by source call-site census only and were not modified, measured or authorized for this slice. Other MT/FKB/TB logits variants would require new resource and bit-identity gates before specialization.
- The full libr3 build, all 140 linked LM metadata comparisons, all 2048 primary/IQR full-byte identity checks, all six actual r3_selftest cases (320 OK), and the two-shape 228-record cold replay were executed. The target M17–32 speed and no-regression gates passed. Static review: only the native logits rewrite and evidence note changed; no vendor, gfx1201 or IQR source edits; `git diff --check` clean.
- Form B was intentionally not built or timed; no Form B divergence statistics are claimed. The E4M3 chained entrypoint is not exposed by the plugin ABI, so only its unchained export was tested. That path and all other nonselected instantiations retain the stock shim.
- **READY for owner review of the Form A int8 M17–32 specialization; not pushed, merged or shipped.** No serving, ms/step A/B or KLD was performed.
