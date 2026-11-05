# rad0012_helper_kernel_config

**Status:** untested
**Plan item:** RSC06

## Summary

Gives the kernel library a mandatory, size-checked, install-once configuration interface, so the helper process hands it the settings it read from the serving manifest instead of the library reading the environment. The engine keeps its own environment read until RSC07.

## Outcome

untested: package and card-free tests exist. The helper's whole cold path, its refusals before any HIP call and the output identity against the previous environment-driven helper were run on hardware with the RSC06 helper (the lane build carrying this patch); the patch has not yet been through a pin-bump gate.

## Detail

The helper configures the plugin before invoking any callback. Shared refusal diagnostics use typed helper settings.

The implementation is instantiated in the upstream row table, which libr4d, libr3 and libr2 already build. Native MoE selection uses the same interface through the BigCherry-owned libr3 rewrite. All upstream edits are managed here.

See [the helper audit](../../components/moe-remote/RSC06_KERNEL_CONFIG.md) for the setting table, the manifest fields each setting comes from and the card-free coverage.
