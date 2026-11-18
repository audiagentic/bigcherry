# GFXOFF kernel recovery investigation

Status: incomplete recovery. Running kernel: `7.0.13-gfxoff`.

The workspace assessment dated November 17 explicitly inventories a `kernel/`
folder containing a Linux 7.0.13 archive, modified source tree, preparation and
diagnostic scripts, and build/install logs. Its disposition was keep pending
patch/provenance checking. That folder is absent on November 18. The installed
kernel's module build link still references its former source tree.

## Observations

- Searched workspace filenames and tracked references, and searched host `/mnt`
  for `linux-7*` directories, gfxoff patch files, and 7.0.13 source archives.
  No original tree, patch or archive was found by these searches. Permission
  errors were suppressed; this is not proof of deletion or a complete backup audit.
- The running kernel uses `CONFIG_MODVERSIONS=y`; headers and symbol versions
  from another kernel must not be treated as its original build artifacts.
- Matching generic 7.0.13 headers exist, but the custom build link is dangling.
- Existing PWR06 records retain R9700 idle measurements and an AMDGPU srcversion;
  these describe observations, not the original patch or a reproducible build.
- No source-control commit matching gfxoff was found by commit-message search.

## Preserved installed artifacts

A host-local recovery bundle is now retained under the workspace-relative
`kernel-recovery/2026-11-18/`. It contains copies of the matching boot files,
including kernel configuration, kernel identity, former build-link target,
AMDGPU module metadata, and SHA-256 checksums for the boot files. It is about
102 MiB. Machine paths remain in that local bundle rather than tracked config.
The bundle is not sufficient to reconstruct the source or `Module.symvers`.
No kernel, bootloader, running service or container was changed.

## Recovery and ownership requirements

Treat the installed kernel, its modules and the recovery bundle as protected
host prerequisites. Do not remove them during cache cleanup. Search backups
and prior build-session records for the original source, patch, compiler
invocation, logs and `Module.symvers`; do not infer the patch from its name.

Once recovered, retain the exact upstream revision/archive hash, complete
ordered patch series, full kernel config, compiler/linker versions and build
commands in source control under a dedicated host-kernel topic. Keep generated
source/build trees under project-local `work/`, with a retained build receipt
and SHA-256 inventory. Preserve generated headers, `Module.symvers`,
`System.map`, kernel image and matching module packages together for every
installed kernel. Cleanup must check installed module build/source links and
bootloader references before admitting any kernel build tree for deletion.

The memory-timing reader remains pending original build-artifact recovery or
an explicitly scheduled boot into a kernel with matching headers. Do not load
an ABI-forced module or replace this kernel to work around missing provenance.
