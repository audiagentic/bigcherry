# VTP37 — gfx1100 IQR drafter NT1 slicing

**Status:** Qualified for owner review; not shipped.

## Change

The gfx1100 libr3 native IQR rewrite replaces the M17–64 rotated i8a8h B4-table GEMM route with successive M16 NT1 tile launches, taking the slice width from one named token-tile constexpr used in both the kernel and launcher. Every slice is a normal R4D_EMIT dispatch, preserving the AQL per-packet chain record. The new route requests no split-K scratch. Other IQR forms and the upstream gfx1201 kernel are unchanged.

A preliminary implementation used the weight form's rot bit to select the route; source inspection showed that the GEMM format does not encode that bit. The compiled final implementation selects the operation's i8a8h dtype and was requalified.

## Validation actually run

- Full gfx1100 libr3 plugin compiled and linked: all 115 component units; all native rewrite anchors matched.
- r3_selftest built and executed on one XTX: gemm_iqr_nt_m64 **1,034 OK, 0 FAIL**; gemm_iqr_gated **36 OK, 0 FAIL**; gemm_iqr_tiled **129 OK, 0 FAIL**. M17/M64 chained and plain B4 tests verified 2 and 4 dispatches and equal outputs.
- Numerical gate on the final compiled plugin and independent CPU FP64 reference: **7 of 7 shapes PASS** over all M17–64, 115,489,152 BF16 outputs per route. New maximum absolute error matches stock for every shape; relative RMS increase is far below the owner's 1e-6 tolerance. Max BF16 ULP remains diagnostic.
- Resource metadata: **840 kernel signatures** in the IQR part-2 unit before/after, zero changes, additions or removals. M16 NT1 kernels have zero register spills and zero private bytes; the M64 NT2 stock forms spill.
- Paired cold replay with rotated weight copies and 256 MiB cache wash: three repetitions, seven shapes, M16/17/20/24/28/32/33/40/48/49/56/64, with chain=0 and chain=1 kernel specializations. All M>16 measurements improve. The five-layer projection-only total including FC once is stock/new **18.3587/4.7398 ms at M24 and 18.3915/4.7665 ms at M32 unchained**, or **18.3936/4.4833 and 18.5506/4.4666 ms** with chained-kernel specialization.

## Limits

These are synthetic kernel-only measurements; no engine serving, end-to-end A/B or KLD was run. The HIP selftest exercises the chained kernel instantiation, not live AQL completion records. AQL source was reviewed: each kernel packet gets its own predecessor target and advances issued count by its workgroups. GPU measurements used short, queued leases; all completed without device errors. Detailed raw tables and commands remain in the external task report.

Owner review and later integration qualification are still required. No push or shipment was performed.

## Controlled four-part resource audit

The IQR compilation family has four parts: p0 740, p1 796, p2 840 and p3 796 kernels (**3,172 total**). All 3,172 kernels have unchanged VGPR, VGPR-spill and private-segment metadata when compared to a matching stock-source build under the same compiler flags. Parts p0/p1/p2 match the cached stock objects directly. Five unrelated p3 prefill kernels initially differed against an older cached object; recompiling unmodified stock p3 source under the candidate build conditions reproduced exactly those five differences. Stock-control versus candidate p3 then matched **796/796** resource triples. These five historical-object differences are therefore not caused by this IQR slicing rewrite.
