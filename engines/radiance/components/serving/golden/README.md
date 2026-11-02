# Golden owned sets (RSC03 slice 2)

`owned_golden.txt` is what the worker's own selection (the code before it moved into `bc_experts`, commit `6a3d11433`)
returns for each case in `golden_cases.h` on the test fixture, for every helper: `case helper ok|refused bytes l:e ...`.
`bc_experts_test` runs the same cases through `bc::experts::select_owned` and compares line for line.

Regenerate (needs the ROCm clang the worker builds with; only on a deliberate change of the selection, with the new
file reviewed against the old one):

    git archive 6a3d11433 engines/radiance/components/moe-remote | tar -x -C "$SCRATCH/main"
    /opt/rocm-7.2.4/lib/llvm/bin/clang++ -x hip -std=gnu++17 -O1 --offload-arch=gfx1100 --no-gpu-bundle-output \
        -DRAD_HAVE_HIP=1 -D__HIP_PLATFORM_AMD__ -I"$SCRATCH/main/engines/radiance/components/moe-remote" \
        -I<radiance libr4d> -I<radiance runtime include/radiance> -Iengines/radiance/components/serving \
        -Iengines/radiance/components/serving/golden golden/gen_golden_main.cpp -o "$SCRATCH/gen_golden_main"
    "$SCRATCH/gen_golden_main" engines/radiance/components/serving/golden/owned_golden.txt "$SCRATCH/fixture"
