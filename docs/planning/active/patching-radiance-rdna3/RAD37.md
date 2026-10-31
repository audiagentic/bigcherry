---
id: RAD37
order: 0
plan: patching-radiance-rdna3
state: completed
created-at: '2026-10-30T10:23:47.000000+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# radiance v1.13.0 -> v1.14.0 bump: re-port libr3 and rad0002, find and fix regressions

## Description

Move the radiance pin from v1.13.0 (6645337) to v1.14.0 (4a61d72, 2026-10-29; 237 files) with libr3 and the engine patches following, and
hold Flash-Next to the v1.13.0 numbers or better. A regression is found with an A/B on the same layout and fixed in this bump, not
accepted.

## Steps

1. pin-bump to v1.14.0 (compile, lane build, CPU smoke pass). Re-anchor the libr3 rules (ar_entry gained a trailing stage argument;
   selftest perf options and form tables moved). Retire R3_HC_NATIVE and R3_HC_COMPARE: experimental, off by default, decode inside the
   noise, and v1.14.0 splits hc_read into three decode forms the flag would have to be carried through.
2. Re-anchor rad0002 (upstream replaced shared_fp32_combine with op_combine in MoeFP8::pass); all five patches clean.
3. A/B on the helper layout at 1M context x 2 sequences, ordered 1.13.0, 1.14.0, 1.14.0, 1.13.0: decode, unique-prompt prefill, per-op profile,
   prefix hits. Decode +1.5%, prefill equal, prefix hit 0.58 s against 0.22 s at 20K (upstream --prefix-resume exact). Fix: the launcher
   sets --prefix-resume nearest (0.22 s / 0.31 s, as on 1.13.0).
4. Gates: Flash-Next KLD (0.0552, 89.75%, identical to 1.13.0), libr3 selftest, 27B and 35B-A3B sanity and speed.

## Detailed Solution & Technical Design

The launcher (tools/lab/radiance/launch-flashnext-best.sh) now serves 1,048,576 tokens (--rope-scaling yarn:4), 2 sequences, 2,048-token chunks
(the QSA score plane is chunk x context / 4: 4 GiB at 4,096 x 1M) and --prefix-resume nearest. One sequence costs 30% of free-text decode.

## Files

engines/radiance/recipes.toml, engines/radiance/components/*/component.toml (built-against), engines/radiance/components/libr3/native/r4d_ar_entry.hip.rw,
r4d_selftest.cpp.rw, native/r4d_hc_bf16.hip.rw (removed), engines/radiance/patches/rad0002_moe_remote_handoff/patch.py, releases/, tools/lab/radiance/launch-flashnext-best.sh,
docs/reference/engines/radiance/models/*.md

## Validation

See Steps 3 and 4 and the v1.14.0 section of docs/reference/engines/radiance/models/qwen3.8-flash-next.md.

## Notes

Follow-ups: 27B drafter verify step (RAD36) may change with v1.14.0's three-wave verify split; upstream now ships libr3d, its own gfx11 kernel
library (claims ^gfx11), to be A/B'd against libr3 as a separate item; RDNA2 expert offload is RCS07 + RHP07.

## Change Log

- 2026-10-30T10:23:47.000000+00:00 (created-by): Created by agent

## Result 2026-10-31

Gates passed: Flash-Next KLD identical to 1.13.0; libr3 selftest 11,095 ok, 0 failures, 191 skipped (tests needing more gfx1201 cards);
27B stream bench equal to 1.13.0 at 1, 2, 4 and 6 streams; 35B-A3B with MTP depth 3 serves on 1.14.0 and returned HTTP 500 on 1.13.0
(decode attention scratch). Launcher default is 1,048,576 tokens, 2 sequences, `--prefix-resume nearest`; Flash-Next decode holds its
92-122 tok/s on natural text to about 290K of used context and falls to 74 at 580K and 50 at 1M. Not measured: the 27B dflash2
drafter (RAD36).

## Validation

Flash-Next KLD (`rad37-fn-kld-10301013`), libr3 selftest (`rad32-r3-selftest-10301021`), A/B decode and prefill (ordered A/B/B/A),
27B stream bench (`rad37-g27-10310140`), 35B-A3B direct decode on both builds.

## Acceptance Criteria

v1.14.0 pinned with its release record; every gate above passed or its gap is stated here; no regression against 1.13.0 on any model
measured.

