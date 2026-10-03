# patch1000: PA35 step 1 narrow hardware-evidence driver

Plan item: PA35 (patch 1000 is migration #8 in PA36's atomic sequence)
Status: active
Owner: PA35 / PA36 (patching-patch-system)
Question state: open

## Question

On `gfx1201` (RDNA4, the patch's target architecture), does patch
`1000_rdna4_mmq_q2k_q6k_fix` produce Q2_K/Q6_K exact-shape `backend-ops`
correctness (full registered `MUL_MAT type_a=q2_K/q6_K, type_b=f32` corpus)
and performance (exact `m=4096, n=512, k=14336`, 5 paired runs) versus the
control (serving-core composition without patch 1000)?

## Inputs

- `--hip-path /opt/rocm --device 2` (gfx1201) on Brutus.
- `run_patch1000_backend_ops_correctness` / `run_patch1000_backend_ops_perf`
  (already-authored backend-ops helpers, in `patch1000_verification.py`
  next to the driver since PA43 moved them out of `validation_campaign.py`).
- Q2_K/Q6_K model-level `llama-bench` lanes are SKIPPED (no fixture
  registered / not executed) — recorded as not executed, not as evidence.

## Outputs

`artifacts/patch1000-pa35-step1.json` — the gfx1201 control-vs-subject
backend-ops correctness + performance results.

## Runtime

GPU required: yes (gfx1201 / device 2)
Real compilation required: yes (control + subject builds)
Mutates canonical BigCherry state: no (one-off driver, per GPT design review
`req_71c1aaa166f446a1`; does not touch `validation_campaign.py`)

## Safety

- One-off hardware-evidence driver, not shared production code and not the
  PA36 patch-local producer-ownership migration.
- Check for live build/lease state before running on a shared tree.

## Disposition

Retained (TRANSITIONAL) as the PA35 step 1 driver; diagnostic hardware
evidence, not a maintained tool.
