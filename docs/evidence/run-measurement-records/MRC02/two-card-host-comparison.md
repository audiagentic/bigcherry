# MRC02: read-only comparison on two real AMD cards

Captured: 2026-11-19 21:41:44 UTC (2026-11-20 local, Australia/Melbourne).
Source: `bigcherry.core.host_facts.snapshot_at_start()`, using the real
`/sys` and `/proc` roots; independent `rocm-smi --showbus --showclocks
--showmaxpower --showmeminfo vram --json` and direct per-card sysfs reads.
No power/clock settings were written and no GPU measurement lease was acquired.
This checks observed collection and flags, not performance or production serving.

| Fact | 0000:03:00.0 | 0000:06:00.0 | Independent check |
| --- | --- | --- | --- |
| BDF | readable | readable | Both `rocm-smi` bus IDs match |
| VRAM total | 25,753,026,560 B | 25,753,026,560 B | `rocm-smi` exact on both |
| MCLK at collection | 1,249 MHz | 1,249 MHz | `rocm-smi` 1,249 MHz for both |
| SCLK at collection | 2,896 MHz | 2,886 MHz | `rocm-smi` later 2,233 / 2,274 MHz; live clocks vary, so not compared as immutable values |
| Current cap `power1_cap` | `0` in sysfs | `0` in sysfs | `rocm-smi` reports package max power `0.0` on both; collector returns **unreadable**, reason `zero_cap_suspect`, never a valid 0 W cap |
| Cap default | 303,000,000 µW | 303,000,000 µW | Independent `power1_cap_default` sysfs |
| Cap min / max | 272,000,000 / 348,000,000 µW | same | Independent `power1_cap_min` / `power1_cap_max` sysfs |
| PCIe card port | 16.0 GT/s, x16 (sysfs) | 16.0 GT/s, x16 (sysfs) | `rocm-smi` shows 16.0 GT/s x8 effective; three upstream sysfs bridge hops are separately recorded, so the card-width reading is not treated as end-to-end bandwidth |
| Per-card firmware and arch | unreadable, reason `unreadable` | unreadable, reason `unreadable` | Corresponding per-card source files unavailable; no cross-card firmware or architecture is invented |

The real-host verification asserted BDF identity, exact VRAM match, numeric
MCLK and a matching `rocm-smi` reading, present PCIe/card/upstream fields,
and the raw zero-cap-to-unreadable classification for both cards. Dynamic
SCLK was reported rather than asserted to stay constant across independent
reads.

The fixture proof remains in `tools/tests/core/test_host_facts.py`, covering
per-field source failures, PCIe upstream, DIMM and MemAvailable readers,
zero-cap behavior, and explicit manifest/record collection.
`tools/tests/core/test_measurement_host_start.py` covers the strict start
receipt, timestamps, new-store refusal, and read-only legacy recovery. Native
`bc_session_host_start_test` exercises publication before a fake engine
child; it does not require a GPU lease.
