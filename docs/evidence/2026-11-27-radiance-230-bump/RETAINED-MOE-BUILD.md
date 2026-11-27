# Retained MoE host build repair

Canonical lane attempt from controller bef0c6238: initial configure failed because the direct invocation omitted the documented FFmpeg prefix. /opt/ffmpeg6 contains the required development libraries and pkg-config metadata; retry with CMAKE_PREFIX_PATH=/opt/ffmpeg6 configured correctly with media support still enabled.

The retry failed before publishing a runtime. Incremental build captured these concrete errors: r4d_fp8_layout.cpp referenced deleted fp8_block_enc and FGeom.sn; r4d_rows.cpp referenced deleted cMoeGemmQ, undeclared r4d_shape_moe_gemm_q and r4d_rad_had_quant_act_fp8. All are retained rad0046 helper contracts, not failures of the retired upstream trunk encoding path.

Repair rad0046: restore exact former fp8a8 constraint (restricted dtype and geometry), explicitly declare callbacks, use the existing affine encoding predicate for the 128x128 BF16-scaled E4M3 planes and derive row scale count. This does not restore retired trunk builders. Keep the existing local helper/overlay contract rather than silently remove its capabilities. Re-materialized the full production composition; GCC14 C++20 syntax checks of r4d_rows.cpp, r4d_fp8_layout.cpp, r4d_layout.cpp and r4d_shapes.cpp all passed with the candidate ABI and recipe SDK include paths. Existing rad0046 exact-pin geometry/idempotent replay passed; the direct fixture uses unittest.SkipTest so it can run without pytest and the executable temporary directory is under the work root (system /tmp is noexec).

Linked rebuild and GPU numerical/quality acceptance are not proven by these checks. No production runtime has been changed.
