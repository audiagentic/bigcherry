# Power: where the watts go and what cuts them (PWR01-PWR04, 2026-10-27)

Layout: radiance engine on the R9700 (gfx1201), one `r3_moe_worker` helper per RX 7900 XTX (gfx1100), the build
server, ROCm 7.2.4, kernel 7.0.13, Intel Core i7-12700KF host (intel_pstate, EPP balance_performance). Plan items:
PWR01-PWR04 (`docs/planning/active/run-power-efficiency/`). Harnesses: `tools/lab/radiance/pwr01-baseline.sh`,
`pwr01-soak.sh`, `pwr02-xtx-sclk-sweep.sh`, `pwr03-ecore-ab.sh`; settings tooling: `bigcherry gpu-power`,
`bigcherry host-check`.

## Baseline (PWR01, run pwr01-baseline-10271712)

Median watts per phase; GPU board power from amd-smi, CPU package from RAPL (read access through a udev rule).

| Device | idle | decode | prefill |
|---|---|---|---|
| XTX helper 0 | 7 | 154 | 198 |
| XTX helper 1 | 8 | 156 | 198 |
| R9700 engine | 20 | 216 | 218 |
| CPU package | 8 | 78 | 82 |
| total (with an unrelated 10 W card) | 53 | 613 | 706 |

Decode 93.5 tok/s (6.6 J/token), prefill 3,312 tok/s (0.21 J/token). Sums of medians, not integrated energy.

## Levers measured

| Lever | Result | Verdict |
|---|---|---|
| XTX power cap | driver minimum 272 W, above the 155-198 W drawn | no effect |
| XTX max shader clock (OD_SCLK) 2800 / 2700 / 2500 / 2000 MHz | decode -2.6 / -1.7 / -2.2 / -7.3%, prefill -1.3 / -2.0 / -2.5 / -7.5%; XTX 118 / 110 / 95 / 72 W in decode | no penalty-free level (decode runs ~3050 MHz); an opt-in power mode at most |
| XTX voltage offset (VDDGFX) at stock clock -25 / -50 / -75 / -100 / -125 mV | XTX decode/prefill W 162/198, 156/192, 149/186, 142/180, 136/175 vs 160/200; speed equal or +1%; decode text hash set identical; no amdgpu errors | **-125 mV: about -24 W per card, no speed cost** (soak below) |
| Blocking HIP completion waits (RAD25) | per-event BlockingSync ignored; device-wide sleeps only after a fixed ~0.3 ms ROCclr+ROCr spin (100 + 200 us constants); same on TheRock 10.1 | no CPU saving at decode cadence without a patched runtime |
| Host callback wake for the large lane (RAD26) | harness: helper CPU 100% -> ~1%, wake +25-30 us, side stream holds the engine stream 4.5 us; graph host nodes fire on every replay | first engine run deadlocked under stream capture; under diagnosis |
| HSA_ENABLE_MWAITX | AMD-only instruction; this CPU is Intel | not applicable |

Prior art agrees (szafranski/rx7900xtx-qwen3.8-27b-tuning: -125 mV raises XTX tok/s at a fixed cap; a clock cap pins
the voltage so an offset stops working under it; per-card stable offsets differ).

## Open

- -125 mV soak (40 min, sustained decode/prefill) - result recorded in PWR02.
- R9700: 210 W cap and its own voltage-offset sweep (each card calibrated separately).
- PWR03: helper processes on the four E-cores vs default placement, by package watts.
- PWR04: settings live per card in the environment config (`[host.*.devices.power]`, with the validating run as
  evidence); `bigcherry gpu-power` applies/restores/reports them; a privileged helper and start-up apply remain.
