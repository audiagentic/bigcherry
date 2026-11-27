# XDP10 — converter TCQ crash isolation

## Stage 1A — forensic and build checkpoint

The archived single-tensor converter trace faults in the HIP runtime at the
`lq::run_rows` launch path (`libquant/lq_tcq_gpu.hip`, near the call to
`hipLaunchKernel`). The faulting OpenMP thread was invoking GPU work;
other `libgomp` threads were waiting. Their frames do not establish a
thread-local stack or trellis-index overrun.

The affected earlier quantizer library and the subsequently installed
quantizer library each contain **only a gfx1201 HIP code object**. The
affected RX 7900 XTX converters used gfx1100. An archived independent
one-kernel probe, compiled for gfx1201 only, produces SIGSEGV (exit -11)
at launch on both gfx1100 cards under ROCm 7.2.4, while succeeding on
gfx1201. A newer ROCm runtime reports a missing device image instead.
This explains why an architecture mismatch can present as a HIP runtime
segfault rather than a recoverable error. No quantizer buffer index is
dereferenced at the fault location; this evidence does **not** prove a
TCQ scratch-array race.

The archived R9700 one-tensor diagnostic reached normal process exit
and was explicitly marked unqualified, rather than SIGSEGV. A separate
R9700 preflight issue with the wrong kernel library is not the target.

A scratch debug build from the release-pinned source completed the
`rad-convert` and `libquant` targets (201 Ninja steps, no failed
subcommands). The converter's host translation units use O1, debug
symbols and frame pointers. Its dual-target quantizer code objects
were enumerated explicitly as gfx1100 and gfx1201. This is an
architecture-coverage comparator, **not yet a passing conversion**.
The matching historical serving component was used in scratch to
keep that pinned source buildable; no installed release or vendor
source was modified.

Still to qualify: ASAN/UBSan build, new OMP thread-count reproduction
matrix, synthetic and full-recipe device conversion. Until those gates
run, XDP10 is investigational and there is no IQ6R production artifact.
