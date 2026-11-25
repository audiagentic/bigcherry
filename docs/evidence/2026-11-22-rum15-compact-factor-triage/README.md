# RUM15 compact-factor prototype triage (2026-11-22)

Existing local R9700 receipts, not rerun here. Prototype is standalone in temporary standalone Radiance source tree, not a committed Radiance patch.

Native correctness: two-factor 35/35, SHA256 011d228912204fdc840420059232c742e041c6825c5351e60ac3e752383b7a36; BF16 gain-hoisted 35/35, SHA256 cc2938bc503059c629c944238270460e8a5f484e28f3a2c9869683c0cba06956.

54 IQ8R timing-only cases per comparison (Qwen fixture, not LTX):
- Two-factor vs stock warm: median +0.804%; M33 N1024 K5120 +28.26%; worst +36.47%.
- Two-factor vs stock rotating: median +0.011%; M33 +7.19%; worst +13.74%.
- Hoisted vs unhoisted factors: median +0.037%; M33 +1.16%; M8 N12288 K5120 +22.25%, stock drift -0.08%. The hoist comparison baseline is NOT production BF16 stock.

Source: temporary benchmark artifact directory: rum15-factor-{r9700,hoist-r9700}-selftest.json and rum15-bench-factor-{warm,rotate,hoist-vs-factor}/comparison.json plus receipt.json.

Next: identify why hoist penalizes M8 N12288; repeat both GPU architectures under leases with controlled A/B/A, validate LTX TP1 shapes and actual activations, then admission/refill/layout-cache gates. No production promotion or shape workaround until confirmed.
