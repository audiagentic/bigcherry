# Candidate C — idle-device retry

**Blocked; not promoted.** A short queued GPU lease was acquired before the unchained baseline IQR selftest. The immediate prelaunch ROCm snapshot showed no KFD PIDs. GPU 0/1 used 27,959,296 of 25,753,026,560 bytes each, GPU 2 used 59,924,480 of 34,208,743,424 bytes, GPU 3 used 17,199,104 of 17,163,091,968 bytes. The first dispatch returned success from the launch wrapper, but HIP then returned error 2 (out of memory), translated to RAD_E_DEVICE (-15). The baseline selftest exited 1 with 144 FAIL lines and zero BF16 output files. This reproduces the earlier failure without pre-existing VRAM contention.

## Differences from passing tests

| Build | Module bytes | Units | Target |
|---|---:|---|---|
| Passing A embedding | 3,650,656 | embedding only | gfx1100 |
| Passing B head | 4,087,640 | vocabulary head only | gfx1100 |
| C diagnostic baseline | 53,786,336 | four IQR partitions | gfx1100 |
| C candidate | 53,303,376 | four IQR partitions | gfx1100 |

All use HIP_VISIBLE_DEVICES=0 and the selftest selects the first visible gfx11 device (ordinal 0); this retry had no ROCR_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES override. No architecture-selection or ordinal change was found in the recorded configurations. The current harness additionally contains scratch IQR-only output and diagnostic hooks, and the baseline module includes the host-side HIP diagnostic. The substantially larger IQR module and different launch path remain possible causes, not proven root causes. The failure after successful dispatch and the free-memory snapshot are insufficient to distinguish HIP module/queue allocation failure from another runtime bookkeeping issue. Earlier restricted relinks also failed, so module size alone is not established as causal.

Per the stop condition, no additional candidate execution, chained identity, IQR pass claim, or replay was attempted after the idle-device baseline failure. The 44 resource-qualified symbols remain unpromoted. The exact saved selftest scratch patch was reverse-applied to remove the temporary edit from the tracked source; committed evidence remains available. No item D work was started.
