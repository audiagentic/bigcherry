# HI168 result

The requested E2E sequence is complete: four single-GPU 9B contrasts followed
by the dual-XTX 27B contrast, each using stock llama.cpp, BC native, and BC
replay with six balanced order blocks.

The production conclusion is parity, not improvement: the dual-XTX 27B BC
native arm is effectively level with stock, and replay is statistically close
but not a repeatable win.  Replay remains candidate/topology-specific; GPU3
shows a prompt-processing loss and GPU2 has an anomalous stock prompt result.
Those cases require follow-up before any global replay promotion.

The physical-device attestation gap is recorded explicitly.  The run set is
therefore suitable for engineering direction and regression triage, but not a
formal decision-grade performance admission.
