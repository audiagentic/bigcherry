# Radiance experiment register

One row per experiment that did not ship as a default, with outcome and reason. Add a row when an experiment branch
is closed or deleted. The detailed notes are linked; the code lives on the named branch or tag until removed.

| Experiment | Item | Outcome | Why | Detail |
|---|---|---|---|---|
| Blocking lanes, GPU-side arrival waits | RAD20 | Rejected | About -14% decode; any pending GPU wait burns about 80 W per XTX (MES fw 0x80). Replaced by burst gating. | `RAD20-helper-waits-2026-10.md`; branch `feat/rad20-blocking-lanes` |
| Large-lane sleep on host callback | RAD26 | Failed on hardware, open | Under HIP stream capture the callback never woke the helper; engine hung. Callback is 25-67 us p50 versus 6-8 us for the spin. | `docs/planning/active/patching-radiance-rdna3/RAD26.md`; branch `feat/rad26-large-lane-wake` |
| Wave-aggregated route compaction in mailbox publish | RAD29 | No result recorded | Branch holds code and an equivalence/timing test; item still pending. | branch `feat/rad29-route-compaction` |
| Helpers on efficiency cores, EPP settings | PWR03 | No result recorded | Harness `pwr03-ecore-ab.sh` written; host is Intel i7-12700KF. | `PWR03.md`; branch `feat/pwr03-cpu-placement` |
| Gated quantise bounded by route count (`LANE_QUANT`) | RAD23 | Shipped via #481 path; #482 closed as superseded | 616 to 28 us a request on the 6900 XT, replay hashes identical. | PR #482 |
| Multi-row packed-f16 W4 decode GEMV (`GEMV_W4MR`) | RAD23 | Closed unmerged (#506), default-off draft | Never validated end to end; needed its own A/B and KLD. | PR #506 |
| Rows-kernel accuracy variants (groupfold, outputlane, rb16, quad-DPP and others) | RAD46 | Quality fixed, speed not | See the note. | `RAD46-rows-kernel-accuracy-2026-11.md`; branches `codex/rad46*` |
| WSL06-A legacy inventory | WSL06 | Superseded | Work-in-progress path inventory; item completed another way. | branch `feat/wsl06a-legacy-inventory` |
