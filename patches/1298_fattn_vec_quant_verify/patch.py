"""1298: HIP flash attention keeps quantized-KV batches of up to 4 queries on the vector kernel.

Without tensor cores (HIP/RDNA) the dispatcher sends quantized-KV attention to the vector kernel only for
Q->ne[1] <= 2; 3+ queries go to the tile kernel, which needs f16 K/V, so launch_fattn converts the WHOLE
q8_0 cache to f16 on every call (dequantize_block_q8_0_f16, ~1 ms/token at 80K for Flash-Next; O(n_kv)).
MTP3 verify is 4 queries, so every target verify step paid that conversion.

The vector kernel handles any query count as ceil(n/2) two-column tiles reading q8_0 directly: for 4 queries
~2 x 1.06 bytes per KV element instead of ~1.06 (read q8) + 2 (write f16) + 2 (read f16). This raises the
quantized-KV vector limit to BIGCHERRY_FA_VEC_QMAX queries (default 4; 2 restores the upstream choice).
"""

import re as _re

from bigcherry.patcher import Edit, FilePatch

GROUP = "core"
STATE = "rejected"

FATTN = FilePatch(
    path="ggml/src/ggml-cuda/fattn.cu",
    language="none",
    description="Quantized-KV attention: vector kernel up to BIGCHERRY_FA_VEC_QMAX queries (default 4).",
    edits=(
        Edit(
            id="fattn-vec-quant-qmax",
            anchor=_re.escape(
                "        } else {\n"
                "            if (Q->ne[1] <= 2) {\n"
                "                return BEST_FATTN_KERNEL_VEC;\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    return BEST_FATTN_KERNEL_TILE;\n"
            ),
            text=(
                "        } else {\n"
                "            // BigCherry 1298: the tile kernel would convert the whole quantized cache to f16 first;\n"
                "            // the vector kernel reads it directly, ceil(n/2) column tiles (MTP verify is 4 queries)\n"
                "            static const int bc_vec_qmax = [] {\n"
                "                const char * e = getenv(\"BIGCHERRY_FA_VEC_QMAX\");\n"
                "                return e != nullptr ? atoi(e) : 4;\n"
                "            }();\n"
                "            if (Q->ne[1] <= std::max(2, bc_vec_qmax)) {\n"
                "                return BEST_FATTN_KERNEL_VEC;\n"
                "            }\n"
                "        }\n"
                "    }\n"
                "    return BEST_FATTN_KERNEL_TILE;\n"
            ),
            mode="replace",
            guard=r"BigCherry 1298: the tile kernel would convert",
            expect_matches=1,
            rationale="The no-tensor-core tail of ggml_cuda_get_best_fattn_kernel (HIP RDNA path); the Turing branch has its own rule.",
        ),
    ),
)

PATCHES = [FATTN]
