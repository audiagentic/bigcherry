"""Generate balanced server A/B configs for the Qwen3.8-27B unsloth quant sweep.

One variable per group: the weight file. Every group shares one binary (@b-3g-control,
multi-arch gfx1100+gfx1201) and an anchor quant so groups can be chained. server-bench "default" reports pp512 and tg128 only. MTP is off so only
the weight-quant kernels differ. Usage: python3 make-configs.py  (writes ./configs/*.json)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = "/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-"
Q = {
    "q8_0": "Q8_0", "q6_k": "UD-Q6_K", "q5_k_m": "UD-Q5_K_M", "q4_k_m": "UD-Q4_K_M",
    "q4_0": "Q4_0", "q4_1": "Q4_1", "iq4_xs": "UD-IQ4_XS", "iq3_xxs": "UD-IQ3_XXS",
    "iq2_xxs": "UD-IQ2_XXS",
}
DEV = {
    "xtx2": ("0,1", ["gfx1100", "gfx1100"], ["0000:03:00.0", "0000:06:00.0"], ["-sm", "tensor"]),
    "r9700": ("2", ["gfx1201"], ["0000:09:00.0"], []),
    "xtx1": ("0", ["gfx1100"], ["0000:03:00.0"], []),
}
GROUPS = {
    "xtx2": [("q8_0", "q6_k", "q5_k_m"), ("q8_0", "q4_k_m", "q4_0"), ("q8_0", "iq4_xs", "iq3_xxs"), ("q8_0", "q4_1", "iq2_xxs")],
    "r9700": [("q8_0", "q6_k", "q5_k_m"), ("q8_0", "q4_k_m", "q4_0"), ("q8_0", "iq4_xs", "iq3_xxs"), ("q8_0", "q4_1", "iq2_xxs")],
    "xtx1": [("q4_k_m", "q6_k", "q5_k_m"), ("q4_k_m", "q4_0", "q4_1"), ("q4_k_m", "iq4_xs", "iq3_xxs"), ("q4_k_m", "iq2_xxs")],
}

out = HERE / "configs"
out.mkdir(exist_ok=True)
for dev, groups in GROUPS.items():
    vis, archs, locs, split = DEV[dev]
    for i, quants in enumerate(groups, 1):
        cfg = {
            "schema_version": 1, "evidence_role": "production", "execution_evidence": "observe",
            "model": M + Q[quants[0]] + ".gguf",
            "server_args": [*split, "-ngl", "99", "--fit", "off", "-c", "8192", "--flash-attn", "on",
                            "--ubatch-size", "512", "--batch-size", "2048", "--threads", "8", "--parallel", "1"],
            # ROCR selects physical GPUs and renumbers them from 0; HIP indices are post-ROCR.
            "environment": {"ROCR_VISIBLE_DEVICES": vis,
                            "HIP_VISIBLE_DEVICES": ",".join(str(i) for i in range(len(vis.split(","))))},
            "expected_execution": {"backend": "ROCm", "architectures": archs, "locators": locs},
            "bench_configs": "default",
            "required_metrics": ["pp512_tps", "tg128_tps"],
            "repetitions": 1,
            "arms": [{"name": q, "mode": "stock", "shutdown_method": "sigint",
                      "binary": "@b-3g-control", "model": M + Q[q] + ".gguf"} for q in quants],
        }
        (out / f"{dev}-g{i}.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
