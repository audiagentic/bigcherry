# Serving host prerequisites: R9700 GFXOFF

The GFX12 compute-idle/GFXOFF kernel change from PWR06 is a **host kernel
prerequisite**, not a power profile or a serving-manifest setting. In stock
7.0.13, the R9700 (gfx1201, PCI ID 0x7551) can idle at about 20.6 W,
3% GPU busy and VDDGFX 612 mV with a HIP/KFD queue resident; with the
GFX12 compute-idle clause removed in 7.0.13-gfxoff (MES 0x81 unchanged),
the same host measured 15.0 W, 0% busy and VDDGFX 29 mV.
These values are historical evidence, **not classifier thresholds**.

## Runtime check and evidence

`tools/bigcherry/hardware/gfxoff.py` samples the **same** engine GPU's
container-visible sysfs (PCI BDF -> amdgpu hwmon) in two phases:
no-process floor immediately before starting the engine, and healthy
engine-loaded but idle **before its first request**. It takes four samples
one second apart in each phase (all engine BDFs sampled in parallel).
It reads `power1_average` in microwatts, `in0_input` VDDGFX in mV,
and `gpu_busy_percent`. No debugfs, AMD SMI, GPU lease or GPU command
is used by the probe. It performs no writes to sysfs.

The judgment is intentionally conservative and **derives all margins from
the card's measured sample ranges**, without watt/mV/busy thresholds.
`absent` requires all three loaded-idle ranges to be strictly above
their floor ranges; `active` requires the loaded-idle ranges to be
a matching floor/loaded spread (both endpoints) on all three sensors.
A wide, noisy floor merely containing a narrower loaded interval is
**inconclusive**, not evidence that GFXOFF is engaged.
Anything else (noise, mixed evidence, missing readings, a driver/sensor
identity change) is `ambiguous` or `unavailable` rather than an
assertion that the patch is active. This is a behavioural proxy,
not an inspection of the kernel source or a direct GFXOFF register.

- Live FlashNext launcher:
  `tools/lab/radiance/launch-flashnext-best.sh` captures
  `runs/.../gfxoff-floor-<STAMP>.json` after helpers and immediately
  before `start_engine_manifest`; it evaluates after `/health` and
  before `start_canary`, writing `gfxoff-<STAMP>.json` in the same
  run folder. Failures never stop serving.
- Python supervisor:
  `tools/bigcherry/serving/supervisor.py` captures just before the
  engine child and after `ProcessLauncher.ready`, before entering the
  request wait loop. Results appear in the supervisor run record and
  in the already-existing PWR07 power receipt's
  `observations.gfxoff` block. The supervisor remains inactive for
  production until the RSC11 serving cutover.
- CLI:
  `bigcherry gfxoff floor --bdf 0000:09:00.0 --output FLOOR.json`
  and, after an idle engine context is resident,
  `bigcherry gfxoff check --bdf 0000:09:00.0 --floor FLOOR.json
  --output RESULT.json`.
  `bigcherry doctor` discovers GFX12 devices and reports
  `needs_idle_engine_context` if no paired observation can be made;
  with a known floor and a deliberately idle HIP/KFD context (which
  may be a minimal test process rather than a server), use
  `bigcherry doctor --gfxoff-bdf 0000:09:00.0 --gfxoff-floor FLOOR.json
  --gfxoff-idle-context`. For offline tests, `--gfxoff-sys-root`
  points at fake sysfs.

The warning for a clear mismatch is:
`WARNING: gfx12 GFXOFF fix not active on <BDF> (idle <W> W, VDDGFX <mV> mV vs floor <W> W, <mV> mV): idle power ~5 W higher; see PWR06`.
When observations agree, it emits `gfxoff: active on <BDF>`;
ambiguous and unavailable results are named distinctly.

The receipt records `uname -r`, `/sys/module/amdgpu/srcversion`,
PCI device ID, and MES/SMC firmware versions **when exposed through sysfs**.
On kernels that expose firmware only in debugfs, the receipt marks each
firmware version `not_exposed_in_sysfs` instead of inventing a value.
No detector infers patch status from the kernel release suffix.

### Measurement preconditions

The floor needs **no process holding that engine card**, and the loaded
phase needs a healthy, request-idle engine that has created and retained its
HIP/KFD compute context. The launcher controls this sequence; a standalone
doctor run cannot silently establish it. The no-process floor must itself show 0% GPU busy; otherwise the probe
refuses to claim the baseline was idle. Other GPU workloads, unstable
sensor data or another process on that card can make the comparison
inconclusive. Re-run with an isolated card rather than turning uncertainty
into a warning that the kernel fix is absent.

Kernel/module changes and MES firmware changes remain owner-run,
rebootable host operations with a separate rollback; this probe never
changes either and never exercises a GPU.
