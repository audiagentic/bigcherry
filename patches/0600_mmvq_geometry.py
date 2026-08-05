"""HI09 - explicit MMVQ geometry variants.

MMVQ is the one family that needs genuinely new compiled code. MMQ, MMVF and MMF
all had their alternatives already compiled behind a runtime switch; MMVQ
derives its geometry from `calc_nwarps` and `calc_rows_per_block` at compile
time, so an alternative geometry is a new template instantiation, not a runtime
argument.

The plan anticipated a large refactor of the kernel body. It turns out not to be
needed. The geometry is consumed through exactly two `constexpr` locals and one
`__launch_bounds__`, so adding two **defaulted** template parameters is enough:

    template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k,
              int nwarps_explicit = 0, int rows_per_block_explicit = 0>

Zero means "derive it as upstream does". Every existing instantiation names
neither parameter and is therefore unchanged, down to the launch bounds -- the
native path is not merely equivalent, it is the same instantiation it was
before. Generated variants name both and get their own.

Standards 14 bounds are `static_assert`ed inside the kernel, so an invalid
geometry fails at compile time rather than reaching a launch. The catalog
applies the same bounds before emitting an instance, so in practice the asserts
should never fire -- they exist to catch a generator bug, which is exactly the
kind of thing that otherwise surfaces as a mysterious runtime failure.
"""

from bigcherry.patcher import Edit, FilePatch

_KERNEL_TEMPLATE = """// bigcherry (HI09): nwarps and rows_per_block become explicit template
// parameters so measured dispatch can compile alternative geometries.
//
// Zero means "derive as upstream does", and both default to zero, so every
// pre-existing instantiation is bit-for-bit the one it was before -- including
// its launch bounds. Only generated variants name them.
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false,
          int nwarps_explicit = 0, int rows_per_block_explicit = 0>
__launch_bounds__((nwarps_explicit != 0 ? nwarps_explicit
                                        : calc_nwarps(type, ncols_dst, get_device_table_id()))
                  *ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q("""

_KERNEL_GEOMETRY = """    constexpr int nwarps = nwarps_explicit != 0
        ? nwarps_explicit
        : calc_nwarps(type, ncols_dst, table_id);
    constexpr int rows_per_cuda_block = rows_per_block_explicit != 0
        ? rows_per_block_explicit
        : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);

    // bigcherry (HI09), standards 14: reject an invalid geometry at compile
    // time. The catalog applies the same bounds before emitting an instance, so
    // these should never fire -- they are here to catch a generator bug, which
    // would otherwise surface as an obscure launch failure.
    static_assert(ncols_dst >= 1 && ncols_dst <= MMVQ_MAX_BATCH_SIZE,
                  "bigcherry: MMVQ width out of bounds");
    static_assert(nwarps >= 1 && nwarps <= 8,
                  "bigcherry: MMVQ nwarps out of bounds");
    static_assert(rows_per_cuda_block >= 1,
                  "bigcherry: MMVQ rows_per_block must be at least 1");"""

_SWITCH_FUSION_TEMPLATE = """// bigcherry (HI09): forwards an explicit geometry to the kernel. Defaulted to
// zero, so existing callers keep the native geometry.
template<ggml_type type, int c_ncols_dst, bool small_k = false,
         int nwarps_explicit = 0, int rows_per_block_explicit = 0>
static void mul_mat_vec_q_switch_fusion("""

_LAUNCH_PARAMS = """// bigcherry (HI09): an explicit geometry changes both the grid and the block
// shape, so the launch parameters must be derived from the same numbers the
// kernel was compiled with. Passing zero keeps upstream's derivation.
template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false,
        const int nwarps_explicit = 0, const int rows_per_block_explicit = 0) {
    const int nwarps = nwarps_explicit != 0
        ? nwarps_explicit
        : calc_nwarps(type, ncols_dst, table_id);
    const int rpb = rows_per_block_explicit != 0
        ? rows_per_block_explicit
        : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);"""

_LAUNCH_INSTANCE = """
#ifdef GGML_HIP_DISPATCH
// bigcherry (HI09): the single bridge from a compiled geometry to a launch.
//
// Generated instances in template-instances/mmvq-autotune-instance-*.cu each
// call this with their own template arguments. The launch parameters are
// derived from the *same* arguments the kernel is instantiated with, which is
// what makes a grid or block shape disagreeing with the compiled geometry
// impossible rather than merely unlikely.
template <ggml_type type, int width, int nwarps, int rows_per_block, bool small_k>
void ggml_hip_mmvq_launch_instance(const ggml_hip_mmvq_launch_args & args) {
    // The no-argument get_device_table_id() is __device__ only; on the host the
    // table is selected from the compute capability.
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);

    const std::pair<dim3, dim3> dims = calc_launch_params<type>(
        width, args.nrows_x, args.nchannels_dst, args.nsamples_or_ntokens,
        args.warp_size, table_id, small_k, nwarps, rows_per_block);

    mul_mat_vec_q_switch_fusion<type, width, small_k, nwarps, rows_per_block>(
        args.vx, args.vy, args.ids, args.fusion, args.dst,
        args.ncols_x, args.nchannels_y, args.stride_row_x, args.stride_col_y,
        args.stride_col_dst, args.channel_ratio, args.stride_channel_x,
        args.stride_channel_y, args.stride_channel_dst, args.sample_ratio,
        args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
        dims.first, dims.second, args.nbytes_shared, args.ids_stride,
        args.stream);
}

// bigcherry (HI09): generated MMVQ geometry instances, and the lookup that
// resolves a forced geometry to one of them.
//
// They live in this translation unit because the templates they instantiate --
// mul_mat_vec_q_switch_fusion and calc_launch_params -- are `static` here.
// Included at this point rather than at the end of the file because
// mul_mat_vec_q_switch_ncols_dst, further down, calls the lookup: everything
// the instances need is defined above, and everything that needs them is below.
#include "hip-autotune-mmvq-instances.inc"
#endif // GGML_HIP_DISPATCH

"""

PATCH = FilePatch(
    path="ggml/src/ggml-cuda/mmvq.cu",
    description="MMVQ explicit geometry as defaulted template parameters",
    edits=(
        Edit(
            id="mmvq-autotune-include",
            anchor=r'^#include "mmvq\.cuh"$',
            rationale="the mmvq.cu include block",
            # Unconditional: ggml_hip_mmvq_forced appears in the signatures of
            # upstream's own switch functions, which are compiled in every
            # build. Everything HIP-specific in the header is guarded inside it.
            text='\n#include "mmvq-autotune.cuh"\n',
            guard=r'#include "mmvq-autotune\.cuh"',
        ),
        Edit(
            id="mmvq-kernel-template",
            anchor=r"^template <ggml_type type, int ncols_dst, bool has_fusion, "
                   r"bool small_k = false>\n"
                   r"__launch_bounds__\(calc_nwarps\(type, ncols_dst, get_device_table_id\(\)\)"
                   r"\*ggml_cuda_get_physical_warp_size\(\), 1\)\n"
                   r"static __global__ void mul_mat_vec_q\(",
            rationale="the MMVQ kernel's template header and launch bounds",
            mode="replace",
            text=_KERNEL_TEMPLATE,
            guard=r"int nwarps_explicit = 0, int rows_per_block_explicit = 0>",
        ),
        Edit(
            id="mmvq-kernel-geometry",
            anchor=r"    constexpr int nwarps = calc_nwarps\(type, ncols_dst, table_id\);\n"
                   r"    constexpr int rows_per_cuda_block = "
                   r"calc_rows_per_block\(ncols_dst, table_id, small_k, nwarps\);",
            rationale="the two constexpr locals the kernel derives its geometry from",
            mode="replace",
            text=_KERNEL_GEOMETRY,
            guard=r"constexpr int nwarps = nwarps_explicit != 0",
        ),
        Edit(
            id="mmvq-launch-params",
            anchor=r"^template<ggml_type type>\n"
                   r"static std::pair<dim3, dim3> calc_launch_params\(\n"
                   r"        const int ncols_dst, const int nrows_x, const int nchannels_dst, "
                   r"const int nsamples_or_ntokens,\n"
                   r"        const int warp_size, const mmvq_parameter_table_id table_id, "
                   r"const bool small_k = false\) \{\n"
                   r"    const int nwarps = calc_nwarps\(type, ncols_dst, table_id\);\n"
                   r"    const int rpb = calc_rows_per_block\(ncols_dst, table_id, small_k, nwarps\);",
            rationale="calc_launch_params, which must agree with the compiled geometry",
            mode="replace",
            text=_LAUNCH_PARAMS,
            guard=r"const int nwarps_explicit = 0, const int rows_per_block_explicit = 0\) \{",
        ),
        Edit(
            id="mmvq-switch-fusion-template",
            anchor=r"^template<ggml_type type, int c_ncols_dst, bool small_k = false>\n"
                   r"static void mul_mat_vec_q_switch_fusion\(",
            rationale="the fusion dispatcher that instantiates the kernel",
            mode="replace",
            text=_SWITCH_FUSION_TEMPLATE,
            guard=r"int nwarps_explicit = 0, int rows_per_block_explicit = 0>\n"
                  r"static void mul_mat_vec_q_switch_fusion\(",
        ),
        Edit(
            id="mmvq-launch-instance",
            # Defined after mul_mat_vec_q_switch_fusion so it can call it, and
            # anchored on the function that follows rather than on the end of
            # the one before -- a closing brace is not a distinctive anchor.
            #
            # This is the single place that turns a compiled geometry into a
            # launch. Deriving the launch parameters from the same template
            # arguments the kernel was instantiated with is what makes a grid
            # and block shape disagreeing with the compiled geometry
            # impossible rather than merely unlikely.
            anchor=r"^template <ggml_type type>\n"
                   r"static void mul_mat_vec_q_moe_launch\(",
            rationale="the function following mul_mat_vec_q_switch_fusion",
            mode="insert_before",
            text=_LAUNCH_INSTANCE,
            guard=r"void ggml_hip_mmvq_launch_instance\(",
        ),
        Edit(
            id="mmvq-kernel-forward",
            # Both kernel instantiations, inside mul_mat_vec_q_switch_fusion.
            anchor=r"mul_mat_vec_q<type, c_ncols_dst, (true|false), small_k>",
            rationale="the two kernel instantiations in the fusion dispatcher",
            mode="replace_all",
            expect_matches=2,
            text=r"mul_mat_vec_q<type, c_ncols_dst, \1, small_k, "
                 r"nwarps_explicit, rows_per_block_explicit>",
            guard=r"small_k, nwarps_explicit, rows_per_block_explicit>",
        ),
    ),
)
