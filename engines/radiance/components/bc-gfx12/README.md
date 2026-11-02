# bc-gfx12

## Summary

BigCherry's RDNA4 kernel component supplies targeted Radiance kernel rows
alongside the upstream library. The engine uses a BigCherry row where its
constraints match and otherwise keeps the upstream implementation.

## Outcome

`BC_GFX12_MOE_GEMV` is marked `validated` and defaults on in
[component.toml](component.toml). Its qualification is scoped to the
recorded MoE model and decode workloads, not to every RDNA4 model or kernel.

## Detail

The current component installs `kernels/libbcgfx12.so`. It includes a
decode-sized FP8 MoE gate/up GEMV path; the down call and larger batches
remain on upstream kernels. The component also contains HyperConn kernel
experiments, whose presence in the source does not establish a production
performance claim.

Read [component.toml](component.toml) for the setting, activation marker and
evidence link. The measured result is in
[the RAD15 evidence bundle](../../../../docs/evidence/RAD15/BC_GFX12_MOE_GEMV.run.json).
