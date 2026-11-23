# Stage 7: grouped resource and dispatch analysis

Groups represent every expert GEMM code object; old/new columns are VGPR count / private bytes / SGPR spill / VGPR spill / VGPR-limited resident waves. Ranges cover other flags, chain variants or M-tile variants sharing the requested group.

| NTL | W4 | ES/AL/RS | FK | Inst. | Stock resource (ranges) | Scaled resource (ranges) | Referenced replay ratios shipped/candidate |
|---:|---:|:---:|---:|---:|:---|:---|:---|
| 4 | 0 | 0/0/0 | 128 | 1 | 112v/560B/3s/307q/9w | 255v/0B/3s/0q/4w | Not exercised |
| 4 | 0 | 0/1/0 | 128 | 1 | 112v/428B/0s/193q/9w | 231v/0B/0s/0q/4w | Not exercised |
| 4 | 0 | 1/0/0 | 128 | 2 | 80v/460B/7s/228q/12w; 80v/464B/12s/229q/12w | 228v/0B/0s/0q/4w; 229v/0B/0s/0q/4w | Not exercised |
| 4 | 0 | 1/0/1 | 128 | 2 | 96v/532B/35s/262q/10w; 96v/536B/36s/265q/10w | 221v/0B/35s/0q/4w; 222v/0B/36s/0q/4w | Not exercised |
| 4 | 1 | 0/0/0 | 128 | 1 | 112v/576B/2s/281q/9w | 256v/0B/2s/0q/4w | Not exercised |
| 4 | 1 | 0/1/0 | 128 | 1 | 112v/388B/0s/159q/9w | 229v/0B/0s/0q/4w | Not exercised |
| 4 | 1 | 1/0/0 | 128 | 2 | 80v/448B/13s/193q/12w; 80v/452B/14s/194q/12w | 228v/0B/0s/0q/4w; 229v/0B/0s/0q/4w | Not exercised |
| 4 | 1 | 1/0/1 | 128 | 2 | 80v/444B/10s/192q/12w; 80v/444B/9s/192q/12w | 228v/0B/0s/0q/4w; 229v/0B/0s/0q/4w | gate_up M4=1.192x; gate_up M8=1.102x; gate_up M16=1.050x; gate_up M32=0.876x!; gate_up M64=0.889x!; other M4=1.361x; other M8=1.118x; other M16=1.055x; other M32=0.877x!; other M64=0.878x! |
| 4 | 2 | 0/0/0 | 128 | 1 | 128v/812B/13s/316q/8w | 256v/312B/13s/102q/4w | Not exercised |
| 4 | 2 | 0/1/0 | 128 | 1 | 112v/740B/0s/247q/9w | 256v/148B/0s/61q/4w | Not exercised |
| 4 | 2 | 1/0/0 | 128 | 2 | 96v/388B/10s/148q/10w; 96v/392B/9s/148q/10w | 225v/0B/0s/0q/4w; 226v/0B/0s/0q/4w | Not exercised |
| 4 | 2 | 1/0/1 | 128 | 2 | 96v/388B/10s/147q/10w; 96v/392B/9s/148q/10w | 225v/0B/0s/0q/4w; 226v/0B/0s/0q/4w | gate_up M16=0.810x!; gate_up M32=0.780x!; gate_up M64=0.772x!; other M16=0.801x!; other M32=0.774x!; other M64=0.770x! |
| 4 | 2 | 1/0/0 | 256 | 2 | 192v/48B/13s/11q/5w; 192v/56B/12s/13q/5w | 247v/0B/0s/0q/4w; 249v/0B/2s/0q/4w | Not exercised |
| 4 | 2 | 1/0/1 | 256 | 2 | 192v/48B/13s/11q/5w; 192v/56B/12s/13q/5w | 247v/0B/0s/0q/4w; 249v/0B/2s/0q/4w | gate_up M1=1.293x; gate_up M4=1.083x; gate_up M8=1.076x; other M1=0.936x!; other M4=1.119x; other M8=1.097x |
| 4 | 3 | 0/0/0 | 128 | 1 | 128v/812B/13s/316q/8w | 256v/312B/13s/91q/4w | Not exercised |
| 4 | 3 | 0/1/0 | 128 | 1 | 112v/740B/0s/251q/9w | 256v/148B/0s/62q/4w | Not exercised |
| 4 | 3 | 1/0/0 | 128 | 2 | 112v/232B/14s/107q/9w; 112v/236B/13s/108q/9w | 209v/0B/0s/0q/4w; 210v/0B/0s/0q/4w | Not exercised |
| 4 | 3 | 1/0/1 | 128 | 2 | 112v/232B/10s/107q/9w; 112v/236B/9s/108q/9w | 209v/0B/0s/0q/4w; 210v/0B/0s/0q/4w | gate_up M16=0.824x!; gate_up M32=0.792x!; gate_up M64=0.795x!; other M16=0.826x!; other M32=0.787x!; other M64=0.787x! |
| 4 | 3 | 1/0/0 | 256 | 2 | 192v/100B/13s/26q/5w; 192v/104B/12s/27q/5w | 219v/0B/13s/0q/4w; 220v/0B/12s/0q/4w | Not exercised |
| 4 | 3 | 1/0/1 | 256 | 2 | 192v/100B/13s/26q/5w; 192v/104B/12s/27q/5w | 219v/0B/13s/0q/4w; 220v/0B/12s/0q/4w | gate_up M1=1.024x; gate_up M4=0.837x!; gate_up M8=0.830x!; other M1=0.731x!; other M4=0.871x!; other M8=0.851x! |
| 4 | 4 | 0/0/0 | 128 | 1 | 128v/840B/13s/327q/8w | 256v/340B/13s/109q/4w | Not exercised |
| 4 | 4 | 0/1/0 | 128 | 1 | 112v/764B/0s/255q/9w | 256v/172B/0s/69q/4w | Not exercised |
| 4 | 4 | 1/0/0 | 128 | 2 | 96v/412B/10s/160q/10w; 96v/416B/9s/160q/10w | 231v/0B/0s/0q/4w; 232v/0B/0s/0q/4w | Not exercised |
| 4 | 4 | 1/0/1 | 128 | 2 | 96v/412B/10s/159q/10w; 96v/416B/9s/160q/10w | 231v/0B/0s/0q/4w; 232v/0B/0s/0q/4w | Not exercised |
| 4 | 4 | 1/0/0 | 256 | 2 | 192v/100B/13s/24q/5w; 192v/108B/12s/26q/5w | 218v/0B/13s/0q/4w; 219v/0B/12s/0q/4w | Not exercised |
| 4 | 4 | 1/0/1 | 256 | 2 | 192v/100B/13s/24q/5w; 192v/108B/12s/26q/5w | 218v/0B/13s/0q/4w; 219v/0B/12s/0q/4w | Not exercised |
| 4 | 5 | 0/0/0 | 128 | 1 | 128v/864B/15s/349q/8w | 256v/364B/15s/129q/4w | Not exercised |
| 4 | 5 | 0/1/0 | 128 | 1 | 112v/788B/0s/263q/9w | 256v/196B/0s/77q/4w | Not exercised |
| 4 | 5 | 1/0/0 | 128 | 2 | 96v/432B/12s/171q/10w; 96v/432B/15s/170q/10w | 237v/0B/0s/0q/4w; 239v/0B/1s/0q/4w | Not exercised |
| 4 | 5 | 1/0/1 | 128 | 2 | 96v/432B/15s/170q/10w; 96v/436B/12s/171q/10w | 237v/0B/0s/0q/4w; 239v/0B/1s/0q/4w | Not exercised |
| 4 | 5 | 1/0/0 | 256 | 2 | 192v/148B/15s/37q/5w; 192v/152B/12s/38q/5w | 230v/0B/12s/0q/4w; 230v/0B/15s/0q/4w | Not exercised |
| 4 | 5 | 1/0/1 | 256 | 2 | 192v/148B/15s/37q/5w; 192v/152B/14s/38q/5w | 230v/0B/14s/0q/4w; 230v/0B/15s/0q/4w | Not exercised |
| 8 | 0 | 0/0/0 | 128 | 1 | 112v/560B/4s/307q/9w | 255v/0B/4s/0q/4w | Not exercised |
| 8 | 0 | 0/1/0 | 128 | 1 | 96v/492B/0s/211q/10w | 231v/0B/0s/0q/4w | Not exercised |
| 8 | 1 | 0/0/0 | 128 | 1 | 112v/576B/3s/281q/9w | 256v/0B/3s/0q/4w | Not exercised |
| 8 | 1 | 0/1/0 | 128 | 1 | 96v/448B/0s/192q/10w | 229v/0B/0s/0q/4w | Not exercised |
| 8 | 2 | 0/0/0 | 128 | 1 | 128v/812B/14s/316q/8w | 256v/312B/14s/102q/4w | Not exercised |
| 8 | 2 | 0/1/0 | 128 | 1 | 112v/740B/0s/247q/9w | 256v/148B/0s/61q/4w | Not exercised |
| 8 | 3 | 0/0/0 | 128 | 1 | 128v/812B/14s/316q/8w | 256v/312B/14s/91q/4w | Not exercised |
| 8 | 3 | 0/1/0 | 128 | 1 | 112v/740B/0s/251q/9w | 256v/148B/0s/62q/4w | Not exercised |
| 8 | 4 | 0/0/0 | 128 | 1 | 128v/840B/14s/327q/8w | 256v/340B/14s/109q/4w | Not exercised |
| 8 | 4 | 0/1/0 | 128 | 1 | 112v/764B/0s/255q/9w | 256v/172B/0s/69q/4w | Not exercised |
| 8 | 5 | 0/0/0 | 128 | 1 | 128v/864B/16s/349q/8w | 256v/364B/16s/129q/4w | Not exercised |
| 8 | 5 | 0/1/0 | 128 | 1 | 112v/788B/0s/263q/9w | 256v/196B/0s/77q/4w | Not exercised |
| 16 | 1 | 0/1/0 | 128 | 1 | 80v/504B/0s/204q/12w | 229v/0B/0s/0q/4w | Not exercised |
| 16 | 2 | 0/1/0 | 128 | 1 | 112v/740B/0s/247q/9w | 256v/148B/0s/62q/4w | Not exercised |
| 16 | 3 | 0/1/0 | 128 | 1 | 112v/740B/0s/250q/9w | 256v/148B/0s/63q/4w | Not exercised |
| 16 | 4 | 0/1/0 | 128 | 1 | 112v/764B/0s/255q/9w | 256v/172B/0s/70q/4w | Not exercised |
| 16 | 5 | 0/1/0 | 128 | 1 | 112v/788B/0s/263q/9w | 256v/196B/0s/78q/4w | Not exercised |

## Dispatch and causal limitations

- The 54-case replay consists of 18 down and 36 token-ordered gate/up or second gate/up cases. All down cases at 1-64 M use a distinct sorted-operand decode kernel; standard W4 M=1 uses a distinct one-row kernel in both gate/up shapes. These timing numbers cannot support an inference about the modified expert GEMM.
- Both token-order geometries use the same ES=1, AL=0, RS=1, NTL=4 expert-GEMM instantiation: W4=1 uses FK=128 for M=4-64; W4=2/3 use FK=256 for M<=8 and FK=128 for M>=16, because T=top_k*M, top_k=10, and the source uses the 80-row FK256 limit.
- An exclamation mark indicates the candidate median slower than the maximum shipped repeat. The table is observational: waves and spills change simultaneously, so the performance loss cannot be uniquely assigned from these two compiled artifacts.
- W4=1 FK128 is faster at M=4/8 but slower at M=32/64 on the very same instantiation in both gate/up geometries. W4=2 FK256 is faster at M=4/8 but the second geometry is slower beyond shipped spread at M=1, on the same instantiation. W4=3 FK256 and both W4=2/3 FK128 have large losses. A static selector of the existing cap cannot retain the demonstrated small-M wins and avoid all measured losses without a further occupancy experiment or runtime/compile-time specialization.
