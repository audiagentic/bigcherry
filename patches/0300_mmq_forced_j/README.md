# 0300 framework validation

This local framework adapter has no Experiment Contract: it checks the emitted
forced-J source transform, not a claimed kernel speedup. The historical
`validated` state is not current qualification.

`mmq-forced-j-transform` compiles and runs the two verbatim emitted blocks
(`_HELPERS` into mmq.cuh, `_MMQ_SOURCE` into mmq.cu) against a minimal host
stub of the CUDA/ROCm surface. It first asserts the compiled bytes equal the
patch edits that carry them, then proves the selected behaviour of those two
blocks: the 16-case `mul_mat_q_launch_forced_J` switch routes to exactly the
J it is given and aborts outside `8..128 step 8`; the lifted native scan
still answers J_best; the PRBE107 `ncols_opt` (not `ncols_max`) delegation
selects the same kernel native picks, including the MoE case where the two
diverge; `mul_mat_q_switch_J` uses a forced value when supplied and the scan
otherwise through one shared launcher; the HI71 dense-shape-aware tail
envelope rejects a forced J outside upstream's own J_max search over the real
batch width, accepts exactly at the envelope boundary, and asserts on a
missing real `ncols_max`; and the `fallback`/`cc` arguments reach the config
tables (the stub carries one fallback-only, cc-gated row that only a
fallback, high-cc caller can see). The patch's remaining forwarding edits --
case signature/forward, instantiation macro, public declaration/definition --
are not exercised by this proof; the real build is what catches those. The
report identifies these as host fixture checks, not completed HIP binary or
GPU architecture evidence.

The universal apply and build checks remain required. Missing source/build
evidence keeps the full adapter ineligible even if the transform check
passes. Real HIP artifacts and identity-bound qualification publication
remain pending the framework execution/evidence integration. No host timing
from these checks answers any server comparison.

Focused offline check: `PYTHONPATH=tools python -m pytest
tools/tests/patch/test_mmq_forced_j_validation_adapter.py
tools/tests/patch/test_framework_validation_packages.py -q` (one shell
command). Raw reports belong in the run directory under
`artifacts/patch-validation/`; only the maintained evidence writer may
publish compact qualification records.
