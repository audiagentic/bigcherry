# RHA11 replay correctness isolation

This evidence isolates the deterministic output divergence found by RHA10 on
the exact-source Qwen3.8-27B dual-XTX production topology. The probe uses the
same server arguments as the RHA04 campaign (`-ngl 99 -sm tensor --fit off
-c 4096 --flash-attn on --ubatch-size 512 --batch-size 2048 --threads 8
--parallel 1`), seed 42, temperature 0, and `n_predict=48`.

The original replay cache selected two tuned MMVQ winners for the first probe
signature set:

* dispatch `f21133b3ee604fd361cb7eb5ccfd1fc3`, signature
  `19a6ba40abbf5e14029d25cf67782eaa`, winner
  `mmvq:q8_0:w1:nw4:rpb4:sk1:v1`;
* dispatch `eb625b97022490d8e6ad5dcaa5cba1b9`, signature
  `a03e95a3b8808c2ebd4160067f73f01e`, winner
  `mmvq:q8_0:w2:nw4:rpb4:sk1:v1`.

Both winners were independently quarantined to `mmvq:native:v1` through the
normal replay-cache export path (seed overrides, not hand-edited runtime
metadata). The resulting cache is `raw/corrected-dispatch.cache`, SHA-256
`A7D9E867F49578C447C99D598FD4200E2612DB007AA5A2DEB4817B9BF872B747`.

The corrected replay was then run against the three-prompt deterministic
corpus used by RHA10. All three content values equal the stock/native corpus:

```
p1 True
p2 True
p3 True
```

This proves the mismatch is caused by the admitted tuned winners, not by the
replay-enabled binary or the production topology. The corrected cache is a
candidate quarantine artifact; RHA10 remains fail-closed until the campaign
cache is regenerated from this seed decision and the diagnostics-off timing
and activation evidence are rerun against that regenerated cache.

Raw response JSON and the manifest-bound seed are retained under `raw/`.
