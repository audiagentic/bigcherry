# 1261 PNRO10 — ctx_other speculative scheduler devices

**State: untested; source placement blocker (2026-10-14).** The patch intends to register missing target-model physical device backends in a separate draft context, after draft-model backends and before ACCEL/CPU. It does not optimize native same-device/tensor-split MTP.

Current `patch.py` inserts after the first `backends.emplace_back(backend);`, which is **inside** the ordinary model-device loop at pinned llama.cpp b11552. Empty draft lists skip foreign devices; overlap with a later ordinary device can duplicate a backend. PNRO10 owns the exact fix and validation. Do not promote 1261 until the anchor is moved outside the loop and source/compile/correctness gates pass.

**Positive activation:** a separate draft/target pair with a missing physical device, `BIGCHERRY_PATCH_TRACE=1`, and observed `BIGCHERRY_PATCH_HIT patch=1261_nro10 path=ctx_other_backend` for the appended device. Same-device MTP produces **no** marker and is a negative control. Backend registration alone does not prove Meta-backed borrowed tensors are schedulable; QFP43/1286 owns the distinct draft-local copy.

See PNRO10 and `TESTING.md`. No current-pin hardware qualification or speedup is established.
