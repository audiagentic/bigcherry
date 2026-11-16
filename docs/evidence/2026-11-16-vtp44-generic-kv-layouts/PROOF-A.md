# Patch A (rad0040) proof: libr4d/r4d_rows.cpp object, before and after

Before = the tree with rad0037 and rad0038 applied; after = plus rad0040. `rows_proof.sh <tree> <prefix>` compiles
`libr4d/r4d_rows.cpp` on the host (g++-14, -O0, no device code) and dumps section bytes (`objdump -s`), relocations (`objdump -r`) and symbols (`nm`).

```
sections  before 54a64a3cb3aea2e0  after 54a64a3cb3aea2e0  lines 30879
relocs  before a91594d45ded182a  after a91594d45ded182a  lines 12449
symbols  before 669546b1fbf100af  after 1b0fdc6f780649b1  lines 1914
```

sections and relocs are identical; symbols differ only in the local names of four static constraint arrays (same addresses):
```
< 0000000000011c80 d cAttnPrefillB16f8
< 0000000000011d60 d cAttnDecodeB16f8
< 0000000000011e60 d cAttnPrefillB16i8
< 0000000000011f40 d cAttnDecodeB16i8
> 0000000000011c80 d cAttnPrefill_gqa6_b16f8
> 0000000000011d60 d cAttnDecode_gqa6_b16f8
> 0000000000011e60 d cAttnPrefill_gqa6_b16i8
> 0000000000011f40 d cAttnDecode_gqa6_b16i8
```
