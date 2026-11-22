# RUM09: the warm-round step time against Radiance 1.17

Question: is the 8-session warm round (prefix-cache hit) about 12% slower per decode step on 2.2 than on 1.17?

Method: the Flash-Next parity harness (`flashnext-4gpu`, production defaults, 262144 context, 8 sessions), one launch per arm, run A B A under one gpu lease each, raw per-round engine step times read from the launch records. A is the installed bc-radiance 1170.36.0 (Radiance 1.17); B is the installed bc-radiance 2020.0.1 (Radiance 2.2). A first 2020.0.1 launch was measured before the arms.

Decode step time, ms (round 0 cold / round 1 warm):

| round | 1.17 A1 | 2.2 B2 | 1.17 A3 | 2.2 first launch |
|---|---|---|---|---|
| decode, 1 stream | 27.0 / 24.7 | 24.6 / 24.4 | 27.5 / 24.4 | 24.4 / 24.3 |
| decode, 6 streams | 48.1 / 46.1 | 46.5 / 47.0 | 47.0 / 48.2 | 47.3 / 46.4 |
| 8 sessions | 28.0 / 58.2 | 28.1 / 59.3 | 28.2 / 59.5 | 28.7 / 56.0 |
| long8 | 30.2 / 25.8 | 25.8 / 27.3 | 30.9 / 27.0 | 25.7 / 27.2 |

Result: 2.2 is within a few percent of 1.17 on every round, including the 8-session warm round (56.0 and 59.3 ms against 58.2 and 59.5 ms). The warm round is about twice the cold round on BOTH builds, so it is not a 2.2 regression. The cause is batch composition: in the warm round the prefix cache lets all eight sessions decode together in every step (eight requests running at once), while in the cold round the long prefills finish one after another and decode runs at five or fewer. A wider step costs more. The release gates (cold step time, decode at 1 and 6 streams, prefill) are met; the warm round stays informational, as the bump guide already says. The earlier +12% is explained by run-to-run spread plus the serving-pool fit fixed in Gate D, and no further investigation is needed.

Harness defect found: `flashnext-parity.sh measure` completes every round and then refuses to publish the run record with "active policy lacks selected calibration store/key" on both builds when the live placement policy is on, so `compare` cannot run. The raw round files above were read directly. This is tracked as a follow-up on the parity harness, not an engine issue.
