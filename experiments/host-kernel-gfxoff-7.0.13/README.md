# Linux 7.0.13-gfxoff — host kernel source reference

This is an **inert, opt-in host-kernel experiment** (plans BHK01–BHK04; evidence `docs/evidence/2026-11-18-gfxoff-kernel-recovery/README.md`). It asks whether an upstream AMD GFX12 compute-queue GFXOFF change can reproduce the observed idle-power improvement and be reconstructed byte-identically. Inputs: pinned Linux 7.0.13 source tarball, exact upstream patch, recovered config and source reference. Outputs: host-local scratch builds, comparisons, and reviewed evidence. Never install or load any module from this archive. Validation limitations and the missing compiler-version match remain recorded below.

This directory records a reproducible **source reconstruction**, not proof of an identical installed binary. It is cataloged as a host-kernel experiment, not a production engine patch, installed module, or release asset. Generated Linux trees, objects, and receipts remain outside Git under `<work-root>/tmp/hostkernel/`.

## Provenance

- Official tarball: https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.0.13.tar.xz (157246100 bytes).
- SHA-256: `3c81edd0f716aca3dd48dff691681827580cc53d35a8eec3be47d346d1f89913`.
- Detached signature: https://cdn.kernel.org/pub/linux/kernel/v7.x/linux-7.0.13.tar.sign. Downloaded, **not cryptographically verified**: GPG invocation was unavailable under the execution policy. The recorded digest is not a substitute for trusted signature verification.
- Upstream commit: `6a1dbb006056ad1ce6df0d6d5f2155767c20f410`, Harkirat Gill, author date 2026-10-21 10:06:36 -0400, committed 2026-10-27 (per PWR06), cherry-picked from `f6d97ecb68b8623da2f83997a9c8c57a1a9649f9`. Commit: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=6a1dbb006056ad1ce6df0d6d5f2155767c20f410
- One established hunk in `drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd.c`: remove the GFX12 clause in `amdgpu_amdkfd_set_compute_idle` while keeping the GFX11 KIQ firmware condition. `series` orders the upstream mail-formatted patch fetched from kernel.org (author/reviewer email addresses redacted to satisfy repository content gates; the diff hunk remains unchanged). The complete patched file is also retained here.
- Base file SHA-256 `336474269575a6625fcdd6e91702f7a3d5b952fe3e3ab57bd7f08cbf3f2d1626`; patched file `89b3f1daa90a62295e489aee21cefc75c626d592570b7c1d7708b2b474026dce`. Both `git apply --check` and `git apply` passed on v7.0.13.

### Ubuntu mainline comparison

The Ubuntu `7.0.13-070013-generic` build dated 2026-06-19 publishes source commit `cc71132ddb23191ff0e02b1ad13fe83ef1c62907`: https://kernel.ubuntu.com/mainline/v7.0.13/ . Its source and diff are available from https://git.launchpad.net/~ubuntu-kernel-test/ubuntu/+source/linux/+git/mainline-crack/ at `cod/mainline/v7.0.13`. The exact `amdgpu_amdkfd.c` retrieved from that commit was **byte-identical** to the upstream 7.0.13 file. Other Ubuntu packaging, config, and source differences were not audited; do not assert the entire Ubuntu tree is identical.

## Configuration and validation

`config-7.0.13-gfxoff` is copied byte-for-byte from the read-only recovery bundle (SHA-256 `43fc0d1c07fa279aa0c9555be3a102a31865ab56eca892a54373f568e8364abe`). It has `CONFIG_LOCALVERSION="-gfxoff"`, `CONFIG_DRM_AMDGPU=m`, `CONFIG_MODVERSIONS=y`, and `CONFIG_MODULE_SRCVERSION_ALL=y`. Recorded build compiler is Ubuntu GCC 13.3.0, GNU binutils 2.42; the attempted rebuild environment has GCC 14.2.0, GNU ld 2.44, Make 4.4.1.

| Module | Recorded srcversion | Rebuilt srcversion |
| --- | --- | --- |
| Stock Ubuntu 7.0.13 | `4F5C8C3B57F8DA15B66C039` | Not obtained |
| Installed 7.0.13-gfxoff | `2F0E6E3215789522FB92997` | Not obtained |

**Build gate:** On 2026-11-18 stock `make O=<scratch>/stock-out olddefconfig` failed at `LEX scripts/kconfig/lexer.lex.c` with `/bin/sh: flex: not found`. `bison` and `bc` were absent too. No package was installed, patched build was not attempted, and no resulting module code or srcversion can be compared. Provision the missing build dependencies and preferably GCC 13.3 with appropriate binutils in a separate permitted build environment. First match the recorded **stock** control hash to validate the method, then attempt to match the patched one; a stock mismatch prevents an exact-reconstruction claim.

Historical PWR06 measurements (R9700, MES 0x81 unchanged, 3 × 60-second repetitions): resident HIP context stock **20.6 W / 3% busy / 612 mV**, patched **15.0 W / 0% busy / 29 mV**. Patched loaded-model idle 15.1 W; after request 15.0 W. These establish behavior, not provenance. The deleted original `_gfxoff_edit*.py` files suggest hand edits; their contents and any additional changed files are **unknown**. No extra patch has been invented.

Repository validation (2026-11-18): forbidden-terms gate passed after temporarily copying and then deleting the local environment config; 545 core/top-level Python tests and 164 subtests passed; ShellCheck and Git whitespace checks passed. No GPU or kernel code was executed.

## Build and compare without installing

From the repository root (substitute an actual work root):

```sh
export BC_WORK_ROOT=<work-root>
bash experiments/host-kernel-gfxoff-7.0.13/build-module.sh '<work-root>/downloads/linux-7.0.13.tar.xz' '<work-root>/tmp/hostkernel-rebuild' 8
mkdir -p '<work-root>/tmp/hostkernel-compare'
bash experiments/host-kernel-gfxoff-7.0.13/compare-release.sh 7.3.5 '<work-root>/tmp/hostkernel-compare'
```

The build script validates the tarball digest and refuses pre-existing scratch directories; extracts separate stock and patched trees; applies the ordered series, checks the retained full file, then runs `olddefconfig`, `modules_prepare`, and only `M=drivers/gpu/drm/amd/amdgpu modules` for each with `nice` and at most 12 make jobs. It reports `modinfo` srcversions where available and retains outputs for review. The comparison script accepts a Linux source directory, tarball, or release version; reports `git apply --check`, target-clause presence, and whether the fix's effect exists. Source inspection does **not** prove commit ancestry.

Neither script installs or loads a kernel/module, runs `depmod`, modifies a bootloader, reboots, invokes `sudo`, or touches system paths. Building a module for actual loading additionally requires ABI-matching `Module.symvers` and kernel build headers. A diagnostic module build is never permission to load it.

## Protected prerequisites and open work

Never delete or modify the installed `7.0.13-gfxoff` kernel, its `/lib/modules/<kernel>` files, boot files, or `<work-root>/kernel-recovery/2026-11-18/` recovery bundle. Before cleanup check both `/lib/modules/<kernel>/build` and `source` symlink targets and all bootloader/kernel references. A dangling build link does not authorize deleting the installed kernel. Delete only individually identified, generated scratch trees after retaining receipts; keep all such trees under the work root.

The memory-timing reader cannot safely be rebuilt/loaded until matching `Module.symvers`, generated headers, and kernel build provenance are recovered or reconstructed. `CONFIG_MODVERSIONS=y`; using a different kernel's headers or forcing module ABI is forbidden. Related evidence: `docs/evidence/2026-11-18-gfxoff-kernel-recovery/README.md`, PWR06 and `docs/reference/serving-host-prerequisites.md`.
