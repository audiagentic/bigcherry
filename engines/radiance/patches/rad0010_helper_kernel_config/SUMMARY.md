# rad0010_helper_kernel_config

RSC06 adds a mandatory size-checked, install-once configuration interface to
the kernel library. The helper configures the plugin before invoking any
callback. Shared refusal diagnostics use typed helper settings; the engine
retains its environment read for RSC07.

The implementation is instantiated in the upstream row table, which libr4d,
libr3 and libr2 already build. Native MoE selection uses the same interface
through the BigCherry-owned libr3 rewrite. All upstream edits are managed here.

See [the helper audit](../../components/moe-remote/RSC06_KERNEL_CONFIG.md)
for the environment-read table, defaults and card-free coverage.
The patch remains untested pending hardware identity validation.
