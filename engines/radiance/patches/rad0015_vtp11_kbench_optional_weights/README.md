# RAD0015 — benchmark-only optional-weight binding correction

State: **untested**, experimental source only. Part of VTP11's numerical
qualification gate. No serving kernel, operator implementation, or model
weights are modified by this patch.

The original Qwen3.8-27B GDN graph declares three fixed weights:
convolution w, A_log and dt_bias. Its kernel schema has a fourth,
**optional convolution-bias slot between w and A_log**. Because this
checkpoint has no bias, the seeded `rad-kbench` fixture represents
that slot as absent. Previously `rad_kbench.cpp` nevertheless advanced
its weight cursor for the missing slot and reported dt_bias missing,
so all 28 gdn_conv_prep cases were skipped while exiting zero.

The patch requires exact agreement between the count of *present*
fixed-weight shape operands and declared fixed-weight handles when
the op has no weight table. It advances its handle cursor only for
present fixed-weight operands. Ambiguous cases skip with an explicit
diagnostic rather than borrowing another weight's bytes. Weight table
paths remain separate and unchanged.

Mechanical gates: 3 focused unit tests including real optional-bias
mapping, fail-closed upstream anchor drift and idempotence. Applied
and re-applied on a scratch copy of pinned Radiance v1.17.0 source.
GCC 14 C++23 syntax check on changed `tools/rad_kbench.cpp` passed.

**Not numerical validation:** Only a real `rad-kbench --op
gdn_conv_prep --kernels libr4d:libref` (or libr3:libref)
with nonzero checked GPU cases and no comparison failures can qualify
the row, and even then the fixture geometry and warm/cold states
must be reviewed against original BF16 serving. Earlier benchmark
attempts from VTP11 produced 0 checked/28 skipped, which is FAIL
despite exit zero.
