# RAD06 libr3 setting evidence

One `ab-benchmark` capture per promoted libr3 setting (lightweight tier, `engines/radiance/components/libr3/component.toml`
`evidence = [...]`). Each run is the R9700 engine plus two RX 7900 XTX helpers on <gpu-host> (the `helper-ab.json` layout,
Flash-Next), two balanced pairs, arms that differ in that one environment variable (the other is on its default), the helper's
startup marker present in the on-arm's log and absent from the control's (`setting_activation: executed`), no failed cell.
`bigcherry patch-lint` re-checks all of that on every run.

| Setting | Config | pp1024 (on vs off) | tg512 (pair ratios) | What it is |
|---|---|---|---|---|
| `R3_MOE_FP16` | `helper-fp16-ab.json` | **+52.4%** (1.522, 1.527) | +2.6% (1.047, 1.007) | grouped MoE GEMM widens E4M3 to f16, f16 WMMA |
| `R3_MOE_AHALF` | `helper-ahalf-ab.json` | +1.2% (1.017, 1.007) | +2.2% (1.043, 1.003) | activations staged as half precision (with FP16) |
| `R3_MOE_HELPER_VECIO` | `helper-vecio-ab.json` | +4.9% (1.048, 1.049) | see note (1.009, 1.631) | 16-byte lane activation I/O |
| `R3_MOE_HELPER_FUSEROUTE` | `helper-fuse-ab.json` | 0.1% (neutral, small lane only) | +3.0% (1.007, 1.054) | fused small-lane front end and routing |
| `R3_MOE_HELPER_FUSESTAGE` | `helper-fusestage-ab.json` | 0.0% (neutral, decode-sized only) | +2.4% (1.039, 1.008) | decode staging folded into the publish kernel |

## How to read these

- Two pairs an arm is a small sample. The performance claim is the direction and the mechanism, not the percentage; the run
  records are `evidence_role: diagnostic` and are not performance-admitted (the harness still wants full source provenance and
  work-equivalence verification). That is the lightweight tier the project uses for promotion; it is not a contract campaign.
- FP16 prefill is the one large effect, and it reproduces the earlier A/B (pp4096 2,130 -> 3,300 tok/s end to end).
- Decode gains are small (about 0.3-5% a pair, all non-negative). Prefill for FUSEROUTE and FUSESTAGE is neutral by design:
  both act on the small (decode-sized) lane only.
- **VECIO decode note.** Pair 2's off arm decoded at 85 / 90 tok/s (tg512 / tg2048) against 137 / 149 in pair 1 while
  generating the same tokens; engine mover summaries and helper stage timings were identical to the normal cell, so the
  slowdown is host-side and intermittent, not the setting. The +63% upper bound and the +28% mean in the record come from
  that cell. Read VECIO decode as the pair-1 ratio (+0.9%), with the slow cell excluded; the prefill result (+4.9%, both pairs
  agree) does not depend on it.
- Output identity: the replay harness gives bitwise-identical results for FP16/AHALF/VECIO/FUSEROUTE/FUSESTAGE against the
  first variant on real captures, and the FP16 end-to-end KL run is unchanged (component.toml summaries).

## Files

`R3_MOE_FP16.run.json`, `R3_MOE_AHALF.run.json`, `R3_MOE_HELPER_VECIO.run.json`, `R3_MOE_HELPER_FUSEROUTE.run.json`,
`R3_MOE_HELPER_FUSESTAGE.run.json`: the `ab-benchmark` run records, copied unchanged from
`${BIGCHERRY_ROOT}/evidence-RAD06/` on <gpu-host> (runs `rad06-evidence-<setting>-<time>`).
