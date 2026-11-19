---
id: BHK01
order: 1
plan: build-host-kernel
state: in_progress
created-at: '2026-11-18T00:45:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Recover and validate the protected Linux 7.0.13-gfxoff host-kernel patch

## Description

The original 7.0.13-gfxoff source tree and GFXOFF edit scripts were lost in the 2026-11-17 work-root purge. Keep the evidenced upstream GFX12 compute-queue GFXOFF fix, source-reference file, recovery config, ordered patch series, build recipe, and cross-release comparison under `host/kernel/7.0.13-gfxoff/`. Generated sources and modules remain in project scratch and never in Git.

## Steps



## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation



## Effort & Risk



## Standards



## Acceptance Criteria



## Notes



## Status 2026-11-19 (step 7 still open)

Step 7 needs two separate confirmations: kernel latency, and long-running serving acceptance. Not done as specified, but there is now serving evidence on the running kernel (`uname -r` = 7.0.13-gfxoff): flashnext-4gpu native starts with the live policy on (a 9-minute warmup run with 14 placement commits, a 2-launch parity run with 256K prefill 3085/3183 tok/s, PR #1179) and a dense TP3 no-regression run the same day (`docs/evidence/2026-11-19-tp3-27b-noregress/` when merged), none with a GPU fault. That is evidence for the serving half; the latency measurement is still to do.

Host checks requested of the user on 2026-11-19 after a random compiler-crash investigation (clang `Segmentation fault` in code generation at about 1 in 160 identical compiles, a GNU `as` junk-byte failure, with 75 GB free, no OOM kills, no ccache involvement, `HardwareCorrupted` 0, a 7.7 TiB memory write-and-verify and CPU check clean): `journalctl -k | grep -iE 'mce|machine check|edac|hardware error|segfault|general protection|out of memory'`, `ras-mc-ctl --errors` or `mcelog --client`, `dmidecode -t memory` (configured speed against rated, XMP/EXPO state), BIOS CPU/memory overclock and curve-optimizer settings, and an overnight memtest86+. EDAC is not exposed in the container and dmesg is unreadable there. Results belong in `docs/evidence/` and in the receipt counts (`<work-root>/build-compile-retries.jsonl`, plan run-measurement-records MRC02).

## Steps and acceptance

1. [x] Record upstream archive URL and SHA-256, downloaded detached signature (cryptographic signature verification remains open), upstream commit identity, and Ubuntu mainline source-commit provenance.
2. [x] Apply one evidenced upstream hunk on clean v7.0.13; verify `git apply --check`; retain full resulting source file and recovered config, without inventing changes attributed to deleted `_gfxoff_edit*.py` scripts.
3. [x] Keep stock/patched amdgpu-only build recipe with 12-job limit, `nice`, and no installation or loading; keep `compare-release.sh` for release regression checks. Follow protected-host cleanup safeguards.
4. [ ] Rebuild stock module first with matching build dependencies, compiler and `CONFIG_MODVERSIONS` evidence; compare to recorded stock `4F5C8C3B57F8DA15B66C039`. Initial attempt stopped at missing `flex` (also missing `bison` and `bc`); no installation authorized.
5. [ ] Rebuild patched module and compare to installed `2F0E6E3215789522FB92997`; if the stock control hash does not match, do not claim exact reproduction. Compare object/code if toolchain permits. Establish signature verification with trusted key and audit Ubuntu source/diff beyond the amdgpu file.
6. [ ] Recover or regenerate **ABI-matching `Module.symvers` and generated headers** for the memory-timing reader; do not reuse generic 7.0.13 headers for the installed custom kernel or force ABI loading.
7. [ ] Confirm kernel latency and long-running serving acceptance separately from the PWR06 idle-power measurement; host remains protected throughout.

## Verification completed

Repository checks: forbidden terms (0 findings), ShellCheck, git whitespace check, and 545 selected Python tests with 164 subtests passed. Kernel module build intentionally stopped at missing flex; bison and bc absent.

## Protected prerequisite and evidence

The installed `7.0.13-gfxoff` kernel, modules, boot records and the read-only `<work-root>/kernel-recovery/2026-11-18/` bundle must not be removed or overwritten. Before cleanup inspect `/lib/modules/<kernel>/build` and `/lib/modules/<kernel>/source` plus bootloader kernel references; a dangling link is not permission to remove the running/installed kernel. All generated source/build trees stay under the work root. Cleanup must name exact scratch trees only and preserve build receipts. Never run kernel installation, `insmod`, `modprobe`, `depmod`, bootloader update, reboot or sudo as part of this plan.

PWR06 measured the R9700 loaded-HIP idle change from stock 20.6 W, 3% busy, 612 mV to patched 15.0 W, 0% busy, 29 mV (MES 0x81 unchanged; three 60-second repetitions). This validates behavior, **not** the full installed source diff. The original `_gfxoff_edit*.py` scripts are missing, so additional historical edits remain uncertain.

Reference: `host/kernel/7.0.13-gfxoff/README.md`; `docs/reference/serving-host-prerequisites.md`; `docs/evidence/2026-11-18-gfxoff-kernel-recovery/README.md`.

## Change Log

- 2026-11-19T07:38:16.846140+00:00 (updated-by): Updated: section:notes
