# 1225_hi85_nccl_heterogeneous_arch_guard: fail closed when a NCCL/RCCL participant lacks PCIe AtomicOps capability (HI85/GP02)

**Status:** untested
**Group:** core
**Plan item:** GP02

## What it does

Adds `ggml_backend_cuda_comm_rccl_admission_ok()` -- a shared, ordinal-independent
admission check queried once per device via the real HIP runtime attribute
`hipDeviceAttributeHostNativeAtomicSupported` -- and calls it before
`ncclCommInitAll` in `ggml_backend_cuda_comm_init_nccl()`, aborting with a
clear, named `GGML_ABORT` (not the uncatchable HIP SIGABRT the crash would
otherwise produce) when any participating device lacks PCIe AtomicOps
completion capability. The admission function is defined once, before every
`comm_init_*` function, specifically so other independent
`ncclCommInitAll()` entry points (patch 0840's hybrid dispatch) can call the
same check.

## Why

GP02 rewrite (2026-09-02): the original version of this guard used raw
GPU-architecture inequality as a proxy for "will this crash RCCL" -- a cheap
heuristic adopted because no better runtime signal was known at the time.
That proxy is now confirmed wrong in a way that actively blocks real work:
it would reject `{0,2}`/`{1,2}` (XTX+R9700), a pair HI138/GP06 spent real
hardware evidence confirming is RCCL-safe, purely because the architectures
differ -- unrelated to the actual failure mechanism.

Real fix: `hipDeviceAttributeHostNativeAtomicSupported` exposes the actual
fact directly, per device, via the standard HIP API. Verified on real
hardware (2026-09-02, standalone HIP probe binary): returns `1` for devices
0/1/2 (all CPU-direct PCIe root ports) and `0` for device 3 (chipset-routed)
on Brutus -- with no device ordinal hardcoded anywhere in the check itself,
consistent with this project's own topology-identity rules (no persistent
identity may be bound to a HIP ordinal/PCI BDF/hostname).

## Real hardware validation of the admission-check mechanism (2026-09-02)

The `hipDeviceAttributeHostNativeAtomicSupported` signal itself (not yet
this patch's full compiled guard) was verified directly against real
hardware via a standalone HIP probe program:

```
device 0 (AMD Radeon RX 7900 XTX): hipDeviceAttributeHostNativeAtomicSupported=1
device 1 (AMD Radeon RX 7900 XTX): hipDeviceAttributeHostNativeAtomicSupported=1
device 2 (AMD Radeon Graphics):    hipDeviceAttributeHostNativeAtomicSupported=1
device 3 (AMD Radeon RX 6900 XT):  hipDeviceAttributeHostNativeAtomicSupported=0
```

Exactly matches HI138's lspci/dmesg-based finding, with a portable runtime
API instead of shelling out to `lspci`/parsing `dmesg`.

## Upstream / provenance

Local design, based on real-hardware findings
(docs/planning/active/hip-autotune/HI85.md, HI138.md). GP02
(docs/planning/active/gpu-collectives/GP02.md) rewrote the trigger from
architecture-inequality to a real per-device PCIe-atomics capability check,
and made the admission logic shared/reusable across every
`ncclCommInitAll()` call site rather than specific to this guard's original
location.
