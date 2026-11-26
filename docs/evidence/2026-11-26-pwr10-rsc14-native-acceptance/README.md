# PWR10 and RSC14: the native managed start under a real power apply and under host-memory contention

Both runs use the installed releases through the verified bundle (`--native-config` copy, so the modified config wins over the bundle's), one lease window each.

## PWR10: first native managed minimum-cap apply (release 2020.21.1, dense qwen27b-tp3)

Profile `efficient` (no rules) selected by `x-bigcherry.power.profile`; per-card values from the work root's `power-site.yaml` (two XTX 272 W, R9700 210 W, their reported minimums). The 6900 XT is never selected: it is only read before and after.

| Step | Result |
|---|---|
| Preconditions: every cap at its reported default (303, 303, 300, 303 W); probe minimums 272, 272, 210 W | pass |
| 1. qualifying apply (`--allow-power-apply --qualify-power`, verified lease) | running; events created, planning, power applied, planned, engine, healthy, canary, running; caps while serving read back 272, 272, 210 W; 6900 XT 303 W |
| 1. stop | exit 0; caps restored exactly to 303, 303, 300 W |
| 1. the run's receipt | apply, restored, 3 applied and 3 restored events, no failure events |
| 2. start without `--qualify-power` | accepted by the receipt match; caps 272, 272, 210 W while serving; restored exactly |
| 3. start with a 215 W R9700 cap (no receipt) | refused (exit 1) naming the missing verified apply, readback and restore receipt; nothing written (caps 303, 303, 300 W) |

A first attempt on 2020.21.0 failed because the bundle's own config shadowed the modified copy (the profile stayed stock); it wrote nothing. The apply path itself then had no end-to-end test: a card-free test now proves that the YAML profile, the site file and the authorisation flag reach the admitted plan, and the three refusals.

## RSC14: host-memory contention (release 2020.22.0, flashnext-4gpu)

| Step | Result |
|---|---|
| a. first start (measures) and stop | running, stop exit 0 |
| b. start while 16 processes leave about 6 GiB of available memory | named early refusal at the engine's pre-weight check, owner exit 97, log names needed and available, stored plan kept, no child of the refused start survives |
| b3. memory released, third start | running on the stored plan, not remeasured |

An earlier attempt on 2020.21.1 found a real defect: under pressure the engine placed a large pinned table on disk, its plan then differed from the admitted one, and the plan check reported a mismatch (exit 2) before the room check, so the owner dropped a good stored plan. The plan check now re-reads a mismatch against the live available memory and reports the typed room refusal when the planned pinned consumers do not fit; a plan that fits stays a mismatch.

Not run, by design: three cold lane builds beside a start and six-stream soaks (the change only alters the owner's reaction to the engine's exit 4), and the three-arm decode and prefill power comparison (not part of the apply path).
