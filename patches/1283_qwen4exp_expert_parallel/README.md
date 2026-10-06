# 1283_qwen4exp_expert_parallel

Promotion record (QFP18 lightweight evidence-reuse tier, pin b11402 / d89651a7). Mechanism is in SUMMARY.md. The
patch is a neutral enabler and stays opt-in (`BIGCHERRY_MOE_EP=1`): with the flag off nothing changes, with the flag
on the tensor split gives each device whole routed experts instead of a row slice of every expert. It is promoted
because the path is proven correct and no slower than the row split, not because it is faster on a model that fits
in VRAM. Its purpose is placement freedom: expert shares apart from `-ts` (`BIGCHERRY_MOE_EP_TS`) and, later, a
host tail for quants whose experts do not fit.

## Evidence

Flash-Next UD-IQ4_XS, production topology (2x RX 7900 XTX gfx1100 + R9700 gfx1201 tensor split, MTP drafter on the
RX 6900 XT), ctx 49152, f16 KV, build `b-moeep-b11402r` (production set + 1281 + 1283),
`tools/lab/flash-next/queue-moe-ep.sh`. ABBA on one binary, A = flag off (row split), B = `BIGCHERRY_MOE_EP=1`.

| Depth | Prefill A (t/s) | Prefill B (t/s) | Decode A (t/s) | Decode B (t/s) |
|---|---|---|---|---|
| 8K | 1040.1 / 1062.1 | 1060.7 / 1082.0 | 84.3 / 86.1 | 85.3 / 86.1 |
| 24K | 1073.5 / 1070.1 | 1079.4 / 1071.4 | 72.5 / 76.0 | 75.2 / 75.7 |

- Flag off is the production build: the greedy text of arm A is the same file (md5 f80ea4df) as the production
  build `b-ixtile-b11402f` on the same prompt and settings, with the same draft acceptance (349/485) and the same
  speed (84.3 - 84.8 t/s there).
- Activation: arm B gives a different greedy text, and its probe distributions differ from the row split while a
  repeat of the row split is identical to itself (TV 0.0000), so the expert-split path ran. The allocation sizes
  follow the expert shares (`BIGCHERRY_MOE_EP_TS=1,1,3` asks the R9700 for 35.6 GB, `1,1,4` for 39.1 GB).
- Correctness with the flag on (`flash-fidelity.sh`, 24 probes, expert split against row split): top-1 21/24,
  TV mean 0.113 at the default shares and 0.078 at shares 1,1,2 - inside the 0.146 envelope of dense GPU against
  CPU f32 on this stack. The expert split is not bit-identical to the row split (different partial sums).
- Work equivalence: draft acceptance 353/472 and 354/469 against 349/485.
- Mechanics: `tools/tests/patch/test_1283_qwen4exp_expert_parallel.py` (apply, idempotent, fail closed) and
  patch-lint. The op it relies on is covered by 1281's reference test (CPU and every GPU batch band).

## Native llama.cpp comparison

Native llama.cpp b11402 has only the row split of the routed experts in its tensor-split backend; the expert-index
split does not exist there. With the flag off this build is the BigCherry production build (same text, same speed),
whose standing against native llama.cpp is recorded with the patches that produce it (native 63K prefill 907.6 t/s
against 1029 for the production set, see 1334's README). With the flag on the speed is equal to that production
row split, so there is no gain or loss against native from this patch by itself.

## What is not claimed

- No speed gain on a model that fits in VRAM. With the R9700 holding half the experts (`BIGCHERRY_MOE_EP_TS=1,1,2`)
  decode was 83.2 / 83.7 against 85.8 / 86.5 t/s (-3%); prefill equal.
- "Dense weights on the XTXs, experts mostly on the R9700" does not load: the routed experts are about 58 GB of this
  file, so the 32 GB card can hold about half of them at most.
- The full 245K context was not fitted with the flag on: the expert split follows `-ts` exactly while the row split
  rounds rows to 128, so a `-ts` tuned for the row split does not fit. Use `BIGCHERRY_MOE_EP_TS` to re-balance.
- No per-device expert-use counters yet, and 1336's selective host copy does not translate ids for range nodes
  (host experts and the expert split are not combined yet).
