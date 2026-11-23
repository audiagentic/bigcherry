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
