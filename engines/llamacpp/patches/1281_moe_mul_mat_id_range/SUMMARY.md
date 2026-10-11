# 1281_moe_mul_mat_id_range

**Status:** validated
**Plan item:** MET02

Kind: enhancement, a new ggml primitive. Ordinary ggml_mul_mat_id is unchanged.

ggml_mul_mat_id_range(ctx, as, b, ids, id_base) is MUL_MAT_ID over a tensor that holds only the experts
[id_base, id_base + as->ne[2]) of a larger set. ids stay global. A lane whose id is in the range is computed with
the local expert; a lane whose id is outside it is an exact +0 and reads no expert weight. A tier graph (MET03) and
whole-expert parallelism (MET04) are sums of such ops, one per device.

The public constructor/accessors and CPU/repack hooks remain in their upstream files. The CUDA implementation is
split by dependency:

- ggml/src/ggml-cuda/bc-moe-range-mmid.cu owns mm_ids_range_translate and
  ggml_cuda_mm_ids_range_translate. This is a separate .cu translation unit because the implementation needs no
  mmid.cu file-local helper; the existing ggml-cuda source glob compiles new .cu files.
- ggml/src/ggml-cuda/bc-moe-range-quantize.cuh owns quantize_scatter_range_mmq_q8_1_cuda. It is included from
  quantize.cu at the old definition site because it instantiates the file-local static quantize_mmq_q8_1 template
  (and its local MMQ layout helper context). No upstream helper is copied.
- ggml/src/ggml-cuda/bc-moe-range-cuda.cuh owns bc_cuda_mul_mat_id_range_supported. It is included immediately
  before ggml_cuda_mul_mat_id_needs_sync because it deliberately uses the dispatch helpers already visible in that
  translation unit. No upstream helper is copied.

common.cuh and mmid.cuh/quantize.cuh retain only the fields/declarations required by callers. mmvq.cu, mmvf.cu,
mmq.cu, quantize.cu and ggml-cuda.cu retain only range translation, sentinel, call and dispatch/fusion hooks.

## Identity proof

Structural identity audit against the pre-refactor 1281 at commit 15e2818127a7873bc9e8cc2795471f1efae5d2fe:
the four moved definitions (the id-translation kernel, its public launcher, the range Q8_1 scatter wrapper, and the
CUDA range-support policy helper) were copied byte-for-byte into the owned files. Their public declarations and all
existing call/dispatch sites are unchanged. The quantize_mmq_q8_1 template and its sentinel edit stay in quantize.cu;
no upstream helper/template was duplicated. Therefore the only compiled-code change is definition placement across
the owned include/translation-unit boundary. A preprocessed-TU diff and compile remain CI/local-build verification,
not evidence claimed by this source audit.

## Evidence

- Offline mechanics test: pending.
- Reference test on Brutus (CPU): pending.
