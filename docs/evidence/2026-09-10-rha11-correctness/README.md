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
replay-enabled binary or the production topology. The complete cache below
extends that quarantine to every previously uncovered native signature, so the
final replay gate can be evaluated without hidden native fallback misses.

Raw response JSON and the manifest-bound seed are retained under `raw/`.

The first corrected cache was also exercised end-to-end with the maintained
`run_bench.py --bench-type server-bench` runner. The diagnostics-on activation
run returned 0 and recorded 21,566/21,566 executed and dispatched operations,
55 cache entries, 33 exact replay matches, 0 unavailable/rerun/incompatible
entries, and positive tuned launches (14,622). That run exposed 24 safe native
fallbacks. Those 24 native-only misses were then added as explicit,
hardware-bound native seed entries through the same exporter. The resulting
79-entry cache is `raw/corrected-dispatch-complete.cache`, SHA-256
`145F849A59C3310171696D38C5049DEB0C39A232A4DE63D963D807817C485C95`.
The final diagnostics-on run returned 0 with 21,566/21,566 dispatches, 57
exact matches, zero misses/unavailable/rerun/incompatible rows, and the final
diagnostics-off production run returned 0 at pp512 927.62, pp2048 1289.19,
tg128 33.71, and tg512 34.05 t/s. The complete cache is now the canonical
Brutus campaign cache; both earlier cache generations remain preserved as
rollback artifacts.
