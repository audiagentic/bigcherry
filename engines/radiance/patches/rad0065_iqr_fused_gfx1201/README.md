# rad0065_iqr_fused_gfx1201

State: **untested**. Stage 1 is a code-only managed native patch.

The gfx1201 native prefill body can read its IQR weight fragments from block-local storage, after each wave decodes one 16-row unit. The native WMMA instructions, integer accumulation and scale-fold behaviour are retained. The constexpr certificate restricts the fused path to computed trellis kind 6, L=16 B2–B7 row-f32, L=15 B2 row-f32, and L=16 B4–B7 bf16 with LRB=0 and LV=0; dispatch requires exactly one prefill band. Every other call uses the prior global fragment-plane route.

Patch mechanics: `python3 -m pytest tools/tests/patch/test_radiance_rad0065.py` and `PYTHONPATH=tools python3 -m bigcherry patch-lint`. Neither is a GPU qualification.

Required build and GPU gates: confirm all instantiated forms compile with the native gfx1201 WMMA; compare decoded fragment-plane bytes and BF16 output bytes, including the ragged supported widths; verify the same wave count, no register spills (the existing plane kernels require 228–240 VGPRs and up to three spills in 25/44 variants), and the reported LDS size within the device limit. Measure qualified single-band shapes and confirm all other forms retain the legacy route. Do not promote without the build, identity, resources and non-regression gates.
