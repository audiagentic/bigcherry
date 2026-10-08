# rd33 (1241) activation evidence

rd33 build 89722ca1/081c8574, Qwen3.8-27B-Q8_0, dual 7900 XTX `-sm tensor`, BIGCHERRY_PATCH_TRACE=1, one 16-token completion, no MTP.
`BIGCHERRY_PATCH_HIT patch=1241_rd33 path=q8_0_f32_decode` fired once per ncols_dst seen: ncols=2, 1, 4 (hits.txt).

CORRECTION to earlier lab notes and REVIEW.md: the shipped gate is `ne1 >= 1 && ne1 <= 8` (PRBE26 widening, patch.py gate anchor),
NOT `ncols_dst == 1`. So rd33 also takes the 5-column MTP verify batch (n_max=4). That explains why MTP acceptance differs
(0.9558 vs 0.9010 in rd33-ab2): the verify pass uses f32 activations. The patch.py docstring still says ncols_dst==1 and is stale.
Consequence: the MTP-lane tg2048 -5.65% is attributable to the widened 2..8 column path (plus different acceptance), and an
ne1==1-only variant is the natural experiment for keeping the plain-decode win without perturbing MTP verify.
Script: ../../activation-check.sh
