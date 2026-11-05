# Radiance in BigCherry

Radiance is BigCherry's primary engine. The pinned **v1.16.0** build
adds AMD RDNA2 and RDNA3 kernel components, remote MoE helpers and narrow
engine patches alongside the RDNA4 engine kernels.

Current documentation:

- [Radiance overview and best results](../../docs/reference/engines/radiance/README.md)
- [Flash-Next](../../docs/reference/engines/radiance/models/qwen3.8-flash-next.md),
  [27B](../../docs/reference/engines/radiance/models/qwen3.8-27b.md) and
  [35B-A3B](../../docs/reference/engines/radiance/models/qwen3.6-35b-a3b.md)
- [Patch and component catalogue](../../docs/reference/engines/radiance/ENHANCEMENTS.md)

The [dated lab history](../../docs/archive/radiance/LAB_HISTORY.md) preserves
previous measurements and investigations. It is not the current performance
or build reference.

## CI documentation changes

Changes confined to Radiance Markdown documentation do not need a native engine
build. The required PR `ci-pass` check still runs; kernel, patch and CMake
changes continue through the normal build and component-test gate.
