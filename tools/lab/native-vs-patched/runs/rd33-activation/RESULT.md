# rd33 (1241) activation evidence

rd33 build 89722ca1/081c8574, Qwen3.8-27B-Q8_0, dual 7900 XTX `-sm tensor`, BIGCHERRY_PATCH_TRACE=1, one 16-token completion, no MTP.
`BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode` fired 3 times (ncols=2, 1, 4; hits.txt).
The ncols=2 and 4 hits occur during warmup/graph-reserve, before/around the decode hit (ncols=1); worth checking that the
patch's ncols gate is not wider than documented (REVIEW.md assumes ncols_dst==1).
Script: ../../activation-check.sh
