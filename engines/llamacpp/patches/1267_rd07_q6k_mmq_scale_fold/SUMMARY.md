# 1267_rd07_q6k_mmq_scale_fold

**Status:** untested; **known_broken on b11552** (not adopted).
**Plan item:** PRBE110; downstream PRBE04.

Hoists Q6_K MMQ row scales and folds sub-scales outside the j0 loop. Historical b11126 gfx1201 pp512 +2.4675% (CI95 [2.2776%,2.6508%]); not a current-pin or validated-state gain.

**Blocker:** b11552/0300 changed Q6_K dispatch to `mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(...)`, invalidating the old `forced_J` marker anchor; the old `mmq.cuh` J-max anchor also disappeared. 0300 already owns preallocation J and HIP `GGML_HIP_MMQ_FORCE_J`. Keep 1267 default-off until PRBE110 proves an eligible Q6_K route and reworks this same package without a duplicate J selector. Preserve PEF01 quarantine and gfx1100/gfx1030 stock fallback.
