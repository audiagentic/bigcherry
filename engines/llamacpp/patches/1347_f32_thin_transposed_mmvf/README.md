# 1347_f32_thin_transposed_mmvf

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7, 2026-10-07). Mechanism is in
SUMMARY.md: an F32 matmul with a 2..8-row weight and more than 8 columns runs through the float vector kernel with the
roles swapped instead of rocBLAS SGEMM. On by default; `BIGCHERRY_F32_THIN_MMVF=0` restores SGEMM.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), ctx 245760, f16 KV, ub512, build `b-metamem-ft2` (production set + 1347). ABBA on one binary with
`tools/lab/flash-next/queue-env-ab.sh`, A = default (on) / B = `BIGCHERRY_F32_THIN_MMVF=0`.

- Gate (prefill kernel profile of the production build, run `gate0-d24576`): the SGEMM kernel serving the 4-row
  hyper-connection projections and the router is 12.7% of an XTX's kernel time, 141 calls per 512-token chunk at
  332 us; 96 of them are the `hc_*_inject` projections (F32 [10240, 4], two per layer).
- Activation (`BIGCHERRY_PATCH_TRACE=1`, run `metamem-ft2`): `k=10240 rows=4 cols=512`.
- Prefill, t/s (complete separation, n = 2 per arm, in every row):

| Run | Depth | Request | Prefill A | Prefill B | Gain |
|---|---|---|---|---|---|
| `ft-ab` | 8K | summarise | 1170.7 / 1172.2 | 1124.8 / 1125.1 | +4.1% |
| `ft-ab` | 24K | summarise | 1175.3 / 1177.0 | 1123.6 / 1125.0 | +4.6% |
| `ft-ab` | 98K | summarise | 1056.1 / 1072.9 | 1033.2 / 1032.1 | +3.1% |
| `ft-ask1` | 24K | ten facts | 1146.3 / 1165.0 | 1105.2 / 1118.9 | +3.9% |
| `ft-ask2` | 24K | critical review | 1171.6 / 1163.8 | 1118.9 / 1122.4 | +4.2% |
| `ft-ask3` | 24K | explain the problem | 1163.1 / 1165.1 | 1111.4 / 1117.2 | +4.5% |
| `ft-plain` (no draft) | 8K | summarise | 1399.0 / 1403.4 | 1302.8 / 1313.6 | +7.1% |
| `ft-plain` (no draft) | 24K | summarise | 1429.6 / 1429.2 | 1339.5 / 1338.1 | +6.8% |

- Accuracy. The sums are formed in another order than SGEMM forms them, so the output is not bit-identical and the
  generated text differs (one md5 per arm in every run). Probes against the CPU f32 reference (`ft-ab`, 24 probes,
  8K): on - top-1 23/24, TV mean 0.0860, max 0.3250; SGEMM - top-1 21/24, TV mean 0.0950, max 0.3902. On against
  SGEMM: top-1 22/24, TV mean 0.0712. Equally far from the reference or closer: stated equivalence, not identity.
- Decode. The path needs more than 8 columns, so it does not run in decode; without a draft, plain decode is equal
  (`ft-plain`: 8K 42.2, 42.1 / 42.2, 42.2 t/s; 24K 39.8, 40.0 / 39.9, 40.2). With the MTP draft the summarise request
  decoded 3-5% slower with the patch (8K 79.7, 80.9 / 83.4, 85.1; 24K 70.4, 71.7 / 74.9, 75.3; 98K 58.9, 58.8 /
  60.6, 60.9) with fewer drafts accepted (e.g. 336-338 against 345-346 of ~500): the three depths share the documents
  and the request, so that is one continuation, not three. On three other requests at 24K the draft acceptance and
  decode are equal: ten facts 73.8, 74.3 / 73.1, 73.7 t/s (343/502 / 343/500); critical review 64.3, 64.0 / 64.0,
  65.5 (320, 318 / 317, 320 accepted); explain 64.5, 65.4 / 63.2, 64.3 (319, 321 / 316, 317). The difference on the
  summarise request is the continuation that request happens to take, not a cost of the patch.
- Offline: package tests (apply, idempotence, placement after upstream's one-row role swap and before the float GEMM,
  fail-closed), patch-lint, production composition check.

## Native llama.cpp comparison

Arm B is native llama.cpp b11402's dispatch for this shape (rocBLAS SGEMM), unmodified, so the ABBAs are native
against the role-swapped vector kernel inside one BigCherry binary: +3.1% to +4.6% prefill with the MTP draft, +6.8%
to +7.1% without. No separate run against a fully native binary was made for this patch.
