"""PRBE41 (AMD-SSM-001): channels-major SSM_CONV input; drop the delta-net transpose.

Port of nasone32/llama.cpp-RDNA3-7900xtx-opt 33611a98 (AMD, Robert Esclapez
Garcia), generated with bigcherry.patch.port_diff and verified byte-exact
against the intended result. ggml_ssm_conv gains an opt-in channels-major
input mode (op_params[0] = 1; input [d_inner, d_conv-1+n_t, n_s], output
layout unchanged), implemented on CPU and CUDA/HIP (short and long-token
kernels); every other backend's supports_op rejects mode != 0 so the
scheduler falls back to CPU. llm_build_delta_net_base::build_conv_state keeps
qkv_mixed channels-major and prepends the recurrent conv state along time,
removing the physical transpose (a CONT kernel) for qwen35 / qwen35moe /
qwen3next. Fork report: pp4096 ub=4096 CONT 221 -> 10 ms.

Port notes:
  * delta-net-base.cpp: the fork's pre-image carried e2188eb2 (a ggml_cont
    this commit deletes) and a different rollback-loop start; the layout
    change is applied to b11126's own text instead, keeping b11126's loop.
  * the Metal backend's supports_op change is NOT ported (it conflicts and is
    never built here); Metal would need the same mode != 0 rejection.
  * the recurrent conv-state memory layout changes from time-major to
    channels-major: states saved by a build without this patch are not
    loadable by one with it (and vice versa).
"""

from bigcherry.patcher import Edit, FilePatch

PROVENANCE = {
    "source-id": "nasone-rdna-optimizations",
    "plan-item": "PRBE41",
    "fork-commit": "33611a98a53af8a327f4a0e01a42631eb5fcd576",
    "port-mode": "port_diff-generated; delta-net-base.cpp hand-merged; Metal excluded",
}

PATCH_01 = FilePatch(
    path='ggml/include/ggml.h',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/include/ggml.h',
    edits=(
        Edit(
            id='prbe41-01-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ bool\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ masked\\);\\\n\\\n\\ \\ \\ \\ GGML_API\\ struct\\ ggml_tensor\\ \\*\\ ggml_ssm_conv\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_context\\ \\*\\ ctx,\\\n',
            text='           bool                  masked);\n\n    // memory layout of the ssm_conv input (sx), stored in op_params[0]\n    enum ggml_ssm_conv_layout {\n        GGML_SSM_CONV_LAYOUT_TIME_MAJOR     = 0, // sx = [d_conv-1+n_t, d_inner, n_s] (time contiguous)\n        GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR = 1, // sx = [d_inner, d_conv-1+n_t, n_s] (channels contiguous)\n    };\n\n    GGML_API struct ggml_tensor * ggml_ssm_conv(\n            struct ggml_context * ctx,\n',
            mode='replace',
            guard='//\\ memory\\ layout\\ of\\ the\\ ssm_conv\\ input\\ \\(sx\\),\\ stored\\ in\\ op_params\\[0\\]',
            rationale='prbe41-01 hunk 1: upstream lines 2524-2523 -> result lines 2524-2529',
            max_span_lines=6,
        ),
        Edit(
            id='prbe41-01-02',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_tensor\\ \\ \\*\\ sx,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_tensor\\ \\ \\*\\ c\\);\\\n\\\n\\ \\ \\ \\ GGML_API\\ struct\\ ggml_tensor\\ \\*\\ ggml_ssm_scan\\(\\\n',
            text='            struct ggml_tensor  * sx,\n            struct ggml_tensor  * c);\n\n    // same as ggml_ssm_conv but sx is channels-major: [d_inner, d_conv-1+n_t, n_s]\n    // (channels contiguous). output layout is unchanged: [d_inner, n_t, n_s].\n    GGML_API struct ggml_tensor * ggml_ssm_conv_channels_major(\n            struct ggml_context * ctx,\n            struct ggml_tensor  * sx,\n            struct ggml_tensor  * c);\n\n    // input (sx) memory layout of an SSM_CONV op (see enum ggml_ssm_conv_layout)\n    GGML_API enum ggml_ssm_conv_layout ggml_ssm_conv_get_layout(const struct ggml_tensor * op);\n\n    GGML_API struct ggml_tensor * ggml_ssm_scan(\n',
            mode='replace',
            guard='//\\ same\\ as\\ ggml_ssm_conv\\ but\\ sx\\ is\\ channels\\-major:\\ \\[d_inner,\\ d_conv\\-1\\+n_t,\\ n_s\\]',
            rationale='prbe41-01 hunk 2: upstream lines 2528-2527 -> result lines 2534-2543',
            max_span_lines=6,
        ),
    ),
)

PATCH_02 = FilePatch(
    path='ggml/src/ggml-backend-meta.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-backend-meta.cpp',
    edits=(
        Edit(
            id='prbe41-02-01',
            anchor='\\\n\\ \\ \\ \\ auto\\ handle_ssm_conv\\ =\\ \\[\\&\\]\\(const\\ std::vector<ggml_backend_meta_split_state>\\ \\&\\ src_ss\\)\\ \\->\\ ggml_backend_meta_split_state\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ if\\ \\(src_ss\\[0\\]\\.axis\\ ==\\ src_ss\\[1\\]\\.axis\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ if\\ \\(src_ss\\[0\\]\\.axis\\ ==\\ GGML_BACKEND_SPLIT_AXIS_0\\)\\ \\{\\\n',
            text='\n    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {\n        if (ggml_ssm_conv_get_layout(tensor) == GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR) {\n            // channels-major: d_inner is sx (src0) axis 0 and c (src1) axis 1, so a d_inner\n            // split lands on different source axes; the output [d_inner, n_t, n_s] splits on axis 0.\n            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1) {\n                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};\n            }\n            return handle_generic(src_ss, /*scalar_only =*/ false);\n        }\n        // time-major: d_inner is axis 1 of both sx and c\n        if (src_ss[0].axis == src_ss[1].axis) {\n            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {\n',
            mode='replace',
            guard='if\\ \\(ggml_ssm_conv_get_layout\\(tensor\\)\\ ==\\ GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR\\)\\ \\{',
            rationale='prbe41-02 hunk 1: upstream lines 802-801 -> result lines 802-810',
            max_span_lines=6,
        ),
    ),
)

PATCH_03 = FilePatch(
    path='ggml/src/ggml-cann/ggml-cann.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-cann/ggml-cann.cpp',
    edits=(
        Edit(
            id='prbe41-03-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_SSM_CONV:\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ true;\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_CUMSUM:\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ op\\->src\\[0\\]\\->type\\ ==\\ GGML_TYPE_F32;\\\n',
            text='            }\n        case GGML_OP_SSM_CONV:\n            // ggml_cann_ssm_conv() requires F32 src/dst and the time-major layout\n            // (the channels-major layout is only implemented on CPU/CUDA)\n            return op->type == GGML_TYPE_F32 &&\n                   op->src[0]->type == GGML_TYPE_F32 &&\n                   op->src[1]->type == GGML_TYPE_F32 &&\n                   ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR;\n        case GGML_OP_CUMSUM:\n            return op->src[0]->type == GGML_TYPE_F32;\n',
            mode='replace',
            guard='//\\ ggml_cann_ssm_conv\\(\\)\\ requires\\ F32\\ src/dst\\ and\\ the\\ time\\-major\\ layout',
            rationale='prbe41-03 hunk 1: upstream lines 2694-2694 -> result lines 2694-2699',
            max_span_lines=7,
        ),
    ),
)

PATCH_04 = FilePatch(
    path='ggml/src/ggml-cpu/ops.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-cpu/ops.cpp',
    edits=(
        Edit(
            id='prbe41-04-01',
            anchor='\\ \\ \\ \\ const\\ int\\ nc\\ \\ =\\ src1\\->ne\\[0\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int\\ ncs\\ =\\ src0\\->ne\\[0\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int\\ nr\\ \\ =\\ src0\\->ne\\[1\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int\\ n_t\\ =\\ \\ dst\\->ne\\[1\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int\\ n_s\\ =\\ \\ dst\\->ne\\[2\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\n\\ \\ \\ \\ GGML_ASSERT\\(\\ dst\\->ne\\[0\\]\\ ==\\ nr\\);\\\n\\ \\ \\ \\ GGML_ASSERT\\(src0\\->nb\\[0\\]\\ ==\\ sizeof\\(float\\)\\);\\\n\\ \\ \\ \\ GGML_ASSERT\\(src1\\->nb\\[0\\]\\ ==\\ sizeof\\(float\\)\\);\\\n\\ \\ \\ \\ GGML_ASSERT\\(src0\\->nb\\[1\\]\\ ==\\ src0\\->ne\\[0\\]\\*sizeof\\(float\\)\\);\\\n',
            text='    //   time-major     : src0 = [d_conv-1+n_t, d_inner, n_s] (time contiguous)\n    //   channels-major : src0 = [d_inner, d_conv-1+n_t, n_s] (channels contiguous)\n    // output is [d_inner, n_t, n_s] (channels contiguous) in both cases\n    const bool channels_major = ggml_ssm_conv_get_layout(dst) == GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR;\n\n    const int nc  = src1->ne[0];                                 // d_conv\n    const int ncs = channels_major ? src0->ne[1] : src0->ne[0];  // d_conv - 1 + n_t\n    const int nr  = channels_major ? src0->ne[0] : src0->ne[1];  // d_inner\n    const int n_t =  dst->ne[1];                                 // tokens per sequence\n    const int n_s =  dst->ne[2];                                 // number of sequences in the batch\n\n    GGML_ASSERT( dst->ne[0] == nr);\n    GGML_ASSERT(src0->nb[0] == sizeof(float));       // input is contiguous (either layout)\n    GGML_ASSERT(src1->nb[0] == sizeof(float));\n    GGML_ASSERT(src0->nb[1] == src0->ne[0]*sizeof(float));\n\n    // byte strides of the input to move one channel / one time step, per layout\n    const size_t chan_nb = channels_major ? src0->nb[0] : src0->nb[1];\n    const size_t time_nb = channels_major ? src0->nb[1] : src0->nb[0];\n    const int    chan_stride = chan_nb / sizeof(float);\n    const int    time_stride = time_nb / sizeof(float);\n    GGML_UNUSED(ncs);\n',
            mode='replace',
            guard='//\\ \\ \\ time\\-major\\ \\ \\ \\ \\ :\\ src0\\ =\\ \\[d_conv\\-1\\+n_t,\\ d_inner,\\ n_s\\]\\ \\(time\\ contiguous\\)',
            rationale='prbe41-04 hunk 1: upstream lines 9712-9721 -> result lines 9712-9733',
            max_span_lines=12,
        ),
        Edit(
            id='prbe41-04-02',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ const\\ float\\ \\*\\ s\\ =\\ \\(const\\ float\\ \\*\\)\\ \\(\\(const\\ char\\ \\*\\)\\ src0\\->data\\ \\+\\ ir0\\*\\(src0\\->nb\\[1\\]\\)\\ \\+\\ i2\\*\\(src0\\->nb\\[0\\]\\)\\ \\+\\ i3\\*\\(src0\\->nb\\[2\\]\\)\\);\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n',
            text='            // sliding window over the time axis (starting at token i2)\n            const float * s = (const float *) ((const char *) src0->data + ir0*chan_nb + i2*time_nb + i3*(src0->nb[2]));\n',
            mode='replace',
            guard='//\\ sliding\\ window\\ over\\ the\\ time\\ axis\\ \\(starting\\ at\\ token\\ i2\\)',
            rationale='prbe41-04 hunk 2: upstream lines 9733-9735 -> result lines 9745-9746',
            max_span_lines=5,
        ),
        Edit(
            id='prbe41-04-03',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ float\\ \\*\\ x\\ =\\ \\(float\\ \\*\\)\\ \\(\\(char\\ \\*\\)\\ dst\\->data\\ \\+\\ ir0\\*\\(dst\\->nb\\[0\\]\\)\\ \\+\\ i2\\*\\(dst\\->nb\\[1\\]\\)\\ \\+\\ i3\\*\\(dst\\->nb\\[2\\]\\)\\);\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ for\\ \\(int\\ i1\\ =\\ 0;\\ i1\\ <\\ ir;\\ \\+\\+i1\\)\\ \\{\\\n',
            text='            float * x = (float *) ((char *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}\n\n            // d_inner\n            for (int i1 = 0; i1 < ir; ++i1) {\n',
            mode='replace',
            guard='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ float\\ \\*\\ x\\ =\\ \\(float\\ \\*\\)\\ \\(\\(char\\ \\*\\)\\ dst\\->data\\ \\+\\ ir0\\*\\(dst\\->nb\\[0\\]\\)\\ \\+\\ i2\\*\\(dst\\->nb\\[1\\]\\)\\ \\+\\ i3\\*\\(dst\\->nb\\[2\\]\\)\\);\\ //\\ \\{d_inner,\\ n_t,\\ n_s\\}\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ //\\ d_inner\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ for\\ \\(int\\ i1\\ =\\ 0;\\ i1\\ <\\ ir;\\ \\+\\+i1\\)\\ \\{\\\n',
            rationale='prbe41-04 hunk 3: upstream lines 9739-9739 -> result lines 9750-9749',
            max_span_lines=7,
        ),
        Edit(
            id='prbe41-04-04',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ sumf\\ \\+=\\ s\\[i0\\ \\+\\ i1\\*ncs\\]\\ \\*\\ c\\[i0\\ \\+\\ i1\\*nc\\];\\\n',
            text='                    sumf += s[i0*time_stride + i1*chan_stride] * c[i0 + i1*nc];\n',
            mode='replace',
            guard='sumf\\ \\+=\\ s\\[i0\\*time_stride\\ \\+\\ i1\\*chan_stride\\]\\ \\*\\ c\\[i0\\ \\+\\ i1\\*nc\\];',
            rationale='prbe41-04 hunk 4: upstream lines 9748-9748 -> result lines 9758-9758',
            max_span_lines=3,
        ),
    ),
)

PATCH_05 = FilePatch(
    path='ggml/src/ggml-cuda/ggml-cuda.cu',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-cuda/ggml-cuda.cu',
    edits=(
        Edit(
            id='prbe41-05-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ op\\->src\\[0\\]\\->ne\\[1\\]\\ %\\ 128\\ ==\\ 0;\\\n',
            text='            // assumes d_inner % threads == 0; d_inner is on ne[1] (time-major) or ne[0] (channels-major)\n            const bool channels_major = ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR;\n            const int64_t d_inner = channels_major ? op->src[0]->ne[0] : op->src[0]->ne[1];\n            return d_inner % 128 == 0;\n',
            mode='replace',
            guard='//\\ assumes\\ d_inner\\ %\\ threads\\ ==\\ 0;\\ d_inner\\ is\\ on\\ ne\\[1\\]\\ \\(time\\-major\\)\\ or\\ ne\\[0\\]\\ \\(channels\\-major\\)',
            rationale='prbe41-05 hunk 1: upstream lines 5486-5487 -> result lines 5486-5489',
            max_span_lines=4,
        ),
    ),
)

PATCH_06 = FilePatch(
    path='ggml/src/ggml-cuda/ssm-conv.cu',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-cuda/ssm-conv.cu',
    edits=(
        Edit(
            id='prbe41-06-01',
            anchor='template\\ <bool\\ apply_silu,\\ size_t\\ split_d_inner,\\ size_t\\ d_conv>\\\n',
            text='// channels_major selects the input (src0) memory layout:\n//   false = time-major     : src0 = [d_conv-1+n_t, d_inner, n_s], time contiguous (nb0)\n//   true  = channels-major : src0 = [d_inner, d_conv-1+n_t, n_s], channels contiguous (nb0)\n// The output (dst) is [d_inner, n_t, n_s] (channels contiguous) in both cases.\ntemplate <bool apply_silu, bool channels_major, size_t split_d_inner, size_t d_conv>\n',
            mode='replace',
            guard='//\\ channels_major\\ selects\\ the\\ input\\ \\(src0\\)\\ memory\\ layout:',
            rationale='prbe41-06 hunk 1: upstream lines 5-5 -> result lines 5-9',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-02',
            anchor='\\ \\ \\ \\ const\\ float\\ \\*\\ GGML_CUDA_RESTRICT\\ bias\\ =\\ bias_ptr;\\\n\\ \\ \\ \\ float\\ \\ \\ \\ \\ \\ \\ \\*\\ GGML_CUDA_RESTRICT\\ dst\\ \\ =\\ dst_ptr;\\\n\\ \\ \\ \\ GGML_UNUSED\\(src0_nb0\\);\\\n\\ \\ \\ \\ const\\ int\\ tid\\ \\ =\\ threadIdx\\.x;\\\n\\ \\ \\ \\ const\\ int\\ bidx\\ =\\ blockIdx\\.x;\\\n',
            text='    const float * GGML_CUDA_RESTRICT bias = bias_ptr;\n    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;\n    const int tid  = threadIdx.x;\n    const int bidx = blockIdx.x;\n',
            mode='replace',
            guard='\\ \\ \\ \\ const\\ float\\ \\*\\ GGML_CUDA_RESTRICT\\ bias\\ =\\ bias_ptr;\\\n\\ \\ \\ \\ float\\ \\ \\ \\ \\ \\ \\ \\*\\ GGML_CUDA_RESTRICT\\ dst\\ \\ =\\ dst_ptr;\\\n\\ \\ \\ \\ const\\ int\\ tid\\ \\ =\\ threadIdx\\.x;\\\n\\ \\ \\ \\ const\\ int\\ bidx\\ =\\ blockIdx\\.x;\\\n',
            rationale='prbe41-06 hunk 2: upstream lines 16-16 -> result lines 20-19',
            max_span_lines=7,
        ),
        Edit(
            id='prbe41-06-03',
            anchor='\\ \\ \\ \\ const\\ float\\ \\*\\ x_block\\ =\\ \\(const\\ float\\ \\*\\)\\ \\(\\(const\\ char\\ \\*\\)\\ src0\\ \\+\\ bidx\\ \\*\\ src0_nb2\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ src0_nb1\\);\\\n',
            text='    // byte strides of the input to move one channel / one time step, per layout\n    const int chan_nb = channels_major ? src0_nb0 : src0_nb1;\n    const int time_nb = channels_major ? src0_nb1 : src0_nb0;\n\n    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * chan_nb);\n',
            mode='replace',
            guard='//\\ byte\\ strides\\ of\\ the\\ input\\ to\\ move\\ one\\ channel\\ /\\ one\\ time\\ step,\\ per\\ layout',
            rationale='prbe41-06 hunk 3: upstream lines 21-21 -> result lines 24-28',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-04',
            anchor='\\ \\ \\ \\ float\\ \\*\\ \\ \\ \\ \\ \\ \\ y_block\\ =\\ \\(float\\ \\*\\)\\ \\(\\(char\\ \\*\\)\\ dst\\ \\+\\ bidx\\ \\*\\ dst_nb2\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ dst_nb0\\);\\\n\\\n\\ \\ \\ \\ const\\ int\\ stride_x\\ =\\ src0_nb1\\ /\\ sizeof\\(float\\);\\\n\\ \\ \\ \\ const\\ int\\ stride_w\\ =\\ src1_nb1\\ /\\ sizeof\\(float\\);\\\n\\ \\ \\ \\ const\\ int\\ stride_y\\ =\\ dst_nb1\\ /\\ sizeof\\(float\\);\\\n\\\n\\ \\ \\ \\ float\\ x\\[d_conv\\]\\ =\\ \\{\\ 0\\.0f\\ \\};\\\n',
            text='    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);\n\n    const int stride_xc = chan_nb / sizeof(float);   // stride to the next channel\n    const int stride_xt = time_nb / sizeof(float);   // stride to the next time step\n    const int stride_w  = src1_nb1 / sizeof(float);\n    const int stride_y  = dst_nb1 / sizeof(float);\n\n    float x[d_conv] = { 0.0f };\n',
            mode='replace',
            guard='const\\ int\\ stride_xc\\ =\\ chan_nb\\ /\\ sizeof\\(float\\);\\ \\ \\ //\\ stride\\ to\\ the\\ next\\ channel',
            rationale='prbe41-06 hunk 4: upstream lines 25-27 -> result lines 32-35',
            max_span_lines=9,
        ),
        Edit(
            id='prbe41-06-05',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ x\\[j\\]\\ =\\ x_block\\[tid\\ \\*\\ stride_x\\ \\+\\ j\\];\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\}\\ else\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ x\\[\\(i\\ \\-\\ 1\\)\\ %\\ d_conv\\]\\ =\\ x_block\\[tid\\ \\*\\ stride_x\\ \\+\\ i\\ \\+\\ d_conv\\ \\-\\ 1\\];\\\n',
            text='                x[j] = x_block[tid * stride_xc + j * stride_xt];\n            }\n        } else {\n            x[(i - 1) % d_conv] = x_block[tid * stride_xc + (i + d_conv - 1) * stride_xt];\n',
            mode='replace',
            guard='x\\[j\\]\\ =\\ x_block\\[tid\\ \\*\\ stride_xc\\ \\+\\ j\\ \\*\\ stride_xt\\];',
            rationale='prbe41-06 hunk 5: upstream lines 45-48 -> result lines 53-56',
            max_span_lines=6,
        ),
        Edit(
            id='prbe41-06-06',
            anchor='template\\ <bool\\ apply_silu,\\ size_t\\ split_d_inner,\\ size_t\\ d_conv,\\ int64_t\\ split_n_t>\\\n',
            text='template <bool apply_silu, bool channels_major, size_t split_d_inner, size_t d_conv, int64_t split_n_t>\n',
            mode='replace',
            guard='template\\ <bool\\ apply_silu,\\ bool\\ channels_major,\\ size_t\\ split_d_inner,\\ size_t\\ d_conv,\\ int64_t\\ split_n_t>',
            rationale='prbe41-06 hunk 6: upstream lines 60-60 -> result lines 68-68',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-07',
            anchor='\\ \\ \\ \\ const\\ float\\ \\*\\ x_block\\ =\\ \\(const\\ float\\ \\*\\)\\ \\(\\(const\\ char\\ \\*\\)\\ src0\\ \\+\\ bidx\\ \\*\\ src0_nb2\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ src0_nb1\\ \\+\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ bidz\\ \\*\\ split_n_t\\ \\*\\ src0_nb0\\);\\\n',
            text='    const int chan_nb = channels_major ? src0_nb0 : src0_nb1;\n    const int time_nb = channels_major ? src0_nb1 : src0_nb0;\n\n    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * chan_nb +\n                                             bidz * split_n_t * time_nb);\n',
            mode='replace',
            guard='const\\ float\\ \\*\\ x_block\\ =\\ \\(const\\ float\\ \\*\\)\\ \\(\\(const\\ char\\ \\*\\)\\ src0\\ \\+\\ bidx\\ \\*\\ src0_nb2\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ chan_nb\\ \\+',
            rationale='prbe41-06 hunk 7: upstream lines 71-72 -> result lines 79-83',
            max_span_lines=4,
        ),
        Edit(
            id='prbe41-06-08',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\(float\\ \\*\\)\\ \\(\\(char\\ \\*\\)\\ dst\\ \\+\\ bidx\\ \\*\\ dst_nb2\\ \\+\\ bidz\\ \\*\\ split_n_t\\ \\*\\ dst_nb1\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ dst_nb0\\);\\\n\\\n\\ \\ \\ \\ const\\ int\\ stride_x\\ =\\ src0_nb1\\ /\\ sizeof\\(float\\);\\\n\\ \\ \\ \\ const\\ int\\ stride_w\\ =\\ src1_nb1\\ /\\ sizeof\\(float\\);\\\n\\ \\ \\ \\ const\\ int\\ stride_y\\ =\\ dst_nb1\\ /\\ sizeof\\(float\\);\\\n\\\n\\ \\ \\ \\ const\\ int64_t\\ local_n_t\\ =\\ min\\(split_n_t,\\ n_t\\ \\-\\ bidz\\ \\*\\ split_n_t\\);\\\n',
            text='        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);\n\n    const int stride_xc = chan_nb / sizeof(float);   // stride to the next channel\n    const int stride_xt = time_nb / sizeof(float);   // stride to the next time step\n    const int stride_w  = src1_nb1 / sizeof(float);\n    const int stride_y  = dst_nb1 / sizeof(float);\n\n    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);\n',
            mode='replace',
            guard='\\ \\ \\ \\ \\ \\ \\ \\ \\(float\\ \\*\\)\\ \\(\\(char\\ \\*\\)\\ dst\\ \\+\\ bidx\\ \\*\\ dst_nb2\\ \\+\\ bidz\\ \\*\\ split_n_t\\ \\*\\ dst_nb1\\ \\+\\ bidy\\ \\*\\ split_d_inner\\ \\*\\ dst_nb0\\);\\\n\\\n\\ \\ \\ \\ const\\ int\\ stride_xc\\ =\\ chan_nb\\ /\\ sizeof\\(float\\);\\ \\ \\ //\\ stride\\ to\\ the\\ next\\ channel\\\n\\ \\ \\ \\ const\\ int\\ stride_xt\\ =\\ time_nb\\ /\\ sizeof\\(float\\);\\ \\ \\ //\\ stride\\ to\\ the\\ next\\ time\\ step\\\n\\ \\ \\ \\ const\\ int\\ stride_w\\ \\ =\\ src1_nb1\\ /\\ sizeof\\(float\\);\\\n\\ \\ \\ \\ const\\ int\\ stride_y\\ \\ =\\ dst_nb1\\ /\\ sizeof\\(float\\);\\\n\\\n\\ \\ \\ \\ const\\ int64_t\\ local_n_t\\ =\\ min\\(split_n_t,\\ n_t\\ \\-\\ bidz\\ \\*\\ split_n_t\\);\\\n',
            rationale='prbe41-06 hunk 8: upstream lines 77-79 -> result lines 88-91',
            max_span_lines=9,
        ),
        Edit(
            id='prbe41-06-09',
            anchor='\\ \\ \\ \\ constexpr\\ int\\ total_elems\\ =\\ split_d_inner\\ \\*\\ load_cols;\\\n\\ \\ \\ \\ int\\ row\\ =\\ tid\\ /\\ load_cols;\\\n\\ \\ \\ \\ int\\ col\\ =\\ tid\\ %\\ load_cols;\\\n\\#pragma\\ unroll\\\n\\ \\ \\ \\ for\\ \\(int\\ idx\\ =\\ 0;\\ idx\\ <\\ total_elems;\\ idx\\ \\+=\\ split_d_inner\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ if\\ \\(row\\ <\\ \\(int\\)split_d_inner\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ smem\\[row\\ \\*\\ n_cols\\ \\+\\ col\\]\\ =\\ x_block\\[row\\ \\*\\ stride_x\\ \\+\\ col\\];\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ col\\ \\+=\\ split_d_inner;\\\n\\ \\ \\ \\ \\ \\ \\ \\ row\\ \\+=\\ col\\ /\\ load_cols;\\\n\\ \\ \\ \\ \\ \\ \\ \\ col\\ \\ =\\ col\\ %\\ load_cols;\\\n\\ \\ \\ \\ \\ \\ \\ \\ if\\ \\(idx\\ >=\\ total_elems\\ \\-\\ tid\\ \\-\\ split_d_inner\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ break;\\\n',
            text='    if (channels_major) {\n        // channels are contiguous: each thread streams its own channel over the\n        // time tile, so consecutive threads read consecutive addresses (coalesced).\n#pragma unroll\n        for (int col = 0; col < load_cols; col++) {\n            smem[tid * n_cols + col] = x_block[tid * stride_xc + col * stride_xt];\n        }\n    } else {\n        // time is contiguous: distribute the [channel, time] tile across threads\n        // so that consecutive threads read consecutive (time) addresses (coalesced).\n        constexpr int total_elems = split_d_inner * load_cols;\n        int row = tid / load_cols;\n        int col = tid % load_cols;\n#pragma unroll\n        for (int idx = 0; idx < total_elems; idx += split_d_inner) {\n            if (row < (int)split_d_inner) {\n                smem[row * n_cols + col] = x_block[row * stride_xc + col * stride_xt];\n            }\n\n            col += split_d_inner;\n            row += col / load_cols;\n            col  = col % load_cols;\n            if (idx >= total_elems - tid - split_d_inner) {\n                break;\n            }\n',
            mode='replace',
            guard='if\\ \\(channels_major\\)\\ \\{',
            rationale='prbe41-06 hunk 9: upstream lines 87-100 -> result lines 99-123',
            max_span_lines=16,
        ),
        Edit(
            id='prbe41-06-10',
            anchor='template\\ <bool\\ apply_silu>\\\n',
            text='template <bool apply_silu, bool channels_major>\n',
            mode='replace',
            guard='template\\ <bool\\ apply_silu,\\ bool\\ channels_major>',
            rationale='prbe41-06 hunk 10: upstream lines 126-126 -> result lines 149-149',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-11',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ggml_cuda_kernel_launch\\(ssm_conv_f32<apply_silu,\\ threads,\\ kNC>,\\ launch_params,\\ src0,\\ src1,\\ bias,\\ src0_nb0,\\ src0_nb1,\\\n',
            text='            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, channels_major, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,\n',
            mode='replace',
            guard='ggml_cuda_kernel_launch\\(ssm_conv_f32<apply_silu,\\ channels_major,\\ threads,\\ kNC>,\\ launch_params,\\ src0,\\ src1,\\ bias,\\ src0_nb0,\\ src0_nb1,',
            rationale='prbe41-06 hunk 11: upstream lines 139-139 -> result lines 162-162',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-12',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ssm_conv_long_token_f32<apply_silu,\\ threads,\\ kNC,\\ split_n_t><<<blocks,\\ threads,\\ smem_size,\\ stream>>>\\(\\\n',
            text='            ssm_conv_long_token_f32<apply_silu, channels_major, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(\n',
            mode='replace',
            guard='ssm_conv_long_token_f32<apply_silu,\\ channels_major,\\ threads,\\ kNC,\\ split_n_t><<<blocks,\\ threads,\\ smem_size,\\ stream>>>\\(',
            rationale='prbe41-06 hunk 12: upstream lines 145-145 -> result lines 168-168',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-06-13',
            anchor='\\ \\ \\ \\ const\\ int64_t\\ nc\\ \\ =\\ src1\\->ne\\[0\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int64_t\\ nr\\ \\ =\\ src0\\->ne\\[1\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int64_t\\ n_t\\ =\\ out\\->ne\\[1\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int64_t\\ n_s\\ =\\ out\\->ne\\[2\\];\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\n\\ \\ \\ \\ GGML_ASSERT\\(out\\->ne\\[0\\]\\ ==\\ nr\\);\\\n\\ \\ \\ \\ GGML_ASSERT\\(src0\\->nb\\[0\\]\\ ==\\ sizeof\\(float\\)\\);\\\n',
            text='    const bool channels_major = ggml_ssm_conv_get_layout(dst) == GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR;\n    // bigcherry PRBE41: activation evidence -- the channels-major input was used.\n    if (channels_major && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {\n        static std::once_flag bigcherry_prbe41_logged;\n        std::call_once(bigcherry_prbe41_logged, [] {\n            GGML_LOG_WARN("BIGCHERRY_PATCH_HIT patch=1263_prbe41 path=ssm_conv_channels_major\\n");\n        });\n    }\n\n    const int64_t nc  = src1->ne[0];                                // d_conv\n    const int64_t nr  = channels_major ? src0->ne[0] : src0->ne[1]; // d_inner\n    const int64_t n_t = out->ne[1];                                 // tokens per sequence\n    const int64_t n_s = out->ne[2];                                 // number of sequences in the batch\n\n    GGML_ASSERT(out->ne[0] == nr);\n    GGML_ASSERT(src0->nb[0] == sizeof(float));      // input is contiguous (either layout)\n',
            mode='replace',
            guard='const\\ bool\\ channels_major\\ =\\ ggml_ssm_conv_get_layout\\(dst\\)\\ ==\\ GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR;',
            rationale='prbe41-06 hunk 13: upstream lines 175-181 -> result lines 198-213',
            max_span_lines=9,
        ),
        Edit(
            id='prbe41-06-14',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ ssm_conv_f32_cuda<true>\\(src0_d,\\ src1_d,\\ bias_d,\\ src0\\->nb\\[0\\],\\ src0\\->nb\\[1\\],\\ src0\\->nb\\[2\\],\\ src1\\->nb\\[1\\],\\ dst_d,\\ out\\->nb\\[0\\],\\ out\\->nb\\[1\\],\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ out\\->nb\\[2\\],\\ nc,\\ nr,\\ n_t,\\ n_s,\\ stream\\);\\\n\\ \\ \\ \\ \\}\\ else\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ ssm_conv_f32_cuda<false>\\(src0_d,\\ src1_d,\\ bias_d,\\ src0\\->nb\\[0\\],\\ src0\\->nb\\[1\\],\\ src0\\->nb\\[2\\],\\ src1\\->nb\\[1\\],\\ dst_d,\\ out\\->nb\\[0\\],\\ out\\->nb\\[1\\],\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ out\\->nb\\[2\\],\\ nc,\\ nr,\\ n_t,\\ n_s,\\ stream\\);\\\n',
            text='        if (channels_major) {\n            ssm_conv_f32_cuda<true, true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],\n                              out->nb[2], nc, nr, n_t, n_s, stream);\n        } else {\n            ssm_conv_f32_cuda<true, false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],\n                              out->nb[2], nc, nr, n_t, n_s, stream);\n        }\n    } else {\n        if (channels_major) {\n            ssm_conv_f32_cuda<false, true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],\n                              out->nb[2], nc, nr, n_t, n_s, stream);\n        } else {\n            ssm_conv_f32_cuda<false, false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],\n                              out->nb[2], nc, nr, n_t, n_s, stream);\n        }\n',
            mode='replace',
            guard='ssm_conv_f32_cuda<true,\\ true>\\(src0_d,\\ src1_d,\\ bias_d,\\ src0\\->nb\\[0\\],\\ src0\\->nb\\[1\\],\\ src0\\->nb\\[2\\],\\ src1\\->nb\\[1\\],\\ dst_d,\\ out\\->nb\\[0\\],\\ out\\->nb\\[1\\],',
            rationale='prbe41-06 hunk 14: upstream lines 200-204 -> result lines 232-246',
            max_span_lines=7,
        ),
    ),
)

PATCH_07 = FilePatch(
    path='ggml/src/ggml-hexagon/ggml-hexagon.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-hexagon/ggml-hexagon.cpp',
    edits=(
        Edit(
            id='prbe41-07-01',
            anchor='\\ \\ \\ \\ const\\ struct\\ ggml_tensor\\ \\*\\ dst\\ \\ =\\ op;\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ if\\ \\(src0\\->type\\ !=\\ GGML_TYPE_F32\\ \\|\\|\\ src1\\->type\\ !=\\ GGML_TYPE_F32\\ \\|\\|\\ dst\\->type\\ !=\\ GGML_TYPE_F32\\)\\ \\{\\\n',
            text='    const struct ggml_tensor * dst  = op;\n\n    // the channels-major input layout is only implemented on CPU/CUDA\n    if (ggml_ssm_conv_get_layout(op) != GGML_SSM_CONV_LAYOUT_TIME_MAJOR) {\n        return false;\n    }\n\n    // Only support FP32 for now\n    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {\n',
            mode='replace',
            guard='//\\ the\\ channels\\-major\\ input\\ layout\\ is\\ only\\ implemented\\ on\\ CPU/CUDA',
            rationale='prbe41-07 hunk 1: upstream lines 5994-5993 -> result lines 5994-5998',
            max_span_lines=6,
        ),
    ),
)

PATCH_08 = FilePatch(
    path='ggml/src/ggml-opencl/ggml-opencl.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-opencl/ggml-opencl.cpp',
    edits=(
        Edit(
            id='prbe41-08-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ \\(op\\->src\\[0\\]\\->type\\ ==\\ GGML_TYPE_F32\\ \\&\\&\\ op\\->src\\[1\\]\\->type\\ ==\\ GGML_TYPE_F32\\ \\&\\&\\ op\\->type\\ ==\\ GGML_TYPE_F32\\);\\\n',
            text='            // the channels-major input layout is only implemented on CPU/CUDA\n            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR);\n',
            mode='replace',
            guard='//\\ the\\ channels\\-major\\ input\\ layout\\ is\\ only\\ implemented\\ on\\ CPU/CUDA',
            rationale='prbe41-08 hunk 1: upstream lines 8919-8919 -> result lines 8919-8920',
            max_span_lines=3,
        ),
    ),
)

PATCH_09 = FilePatch(
    path='ggml/src/ggml-sycl/ggml-sycl.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-sycl/ggml-sycl.cpp',
    edits=(
        Edit(
            id='prbe41-09-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ op\\->type\\ ==\\ GGML_TYPE_F32\\ \\&\\&\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ op\\->src\\[0\\]\\->type\\ ==\\ GGML_TYPE_F32\\ \\&\\&\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ op\\->src\\[1\\]\\->type\\ ==\\ GGML_TYPE_F32;\\\n',
            text='            // the channels-major input layout is only implemented on CPU/CUDA\n            return op->type == GGML_TYPE_F32 &&\n                   op->src[0]->type == GGML_TYPE_F32 &&\n                   op->src[1]->type == GGML_TYPE_F32 &&\n                   ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR;\n',
            mode='replace',
            guard='//\\ the\\ channels\\-major\\ input\\ layout\\ is\\ only\\ implemented\\ on\\ CPU/CUDA',
            rationale='prbe41-09 hunk 1: upstream lines 6786-6788 -> result lines 6786-6790',
            max_span_lines=5,
        ),
    ),
)

PATCH_10 = FilePatch(
    path='ggml/src/ggml-vulkan/ggml-vulkan.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-vulkan/ggml-vulkan.cpp',
    edits=(
        Edit(
            id='prbe41-10-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_SSM_CONV:\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ op\\->src\\[0\\]\\->type\\ ==\\ GGML_TYPE_F32;\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_CONV_TRANSPOSE_1D:\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ return\\ op\\->src\\[0\\]\\->type\\ ==\\ GGML_TYPE_F32\\ \\&\\&\\ op\\->src\\[1\\]\\->type\\ ==\\ GGML_TYPE_F32;\\\n',
            text='            }\n        case GGML_OP_SSM_CONV:\n            // the channels-major input layout is only implemented on CPU/CUDA\n            return op->src[0]->type == GGML_TYPE_F32 && ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR;\n        case GGML_OP_CONV_TRANSPOSE_1D:\n            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;\n',
            mode='replace',
            guard='//\\ the\\ channels\\-major\\ input\\ layout\\ is\\ only\\ implemented\\ on\\ CPU/CUDA',
            rationale='prbe41-10 hunk 1: upstream lines 15640-15640 -> result lines 15640-15641',
            max_span_lines=7,
        ),
    ),
)

PATCH_11 = FilePatch(
    path='ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    edits=(
        Edit(
            id='prbe41-11-01',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ break;\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_SSM_CONV:\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ supports_op\\ =\\ op\\->type\\ ==\\ GGML_TYPE_F32;\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ break;\\\n\\ \\ \\ \\ \\ \\ \\ \\ case\\ GGML_OP_SSM_SCAN:\\\n',
            text='            break;\n        case GGML_OP_SSM_CONV:\n            // the channels-major input layout is only implemented on CPU/CUDA\n            supports_op = op->type == GGML_TYPE_F32 && ggml_ssm_conv_get_layout(op) == GGML_SSM_CONV_LAYOUT_TIME_MAJOR;\n            break;\n        case GGML_OP_SSM_SCAN:\n',
            mode='replace',
            guard='//\\ the\\ channels\\-major\\ input\\ layout\\ is\\ only\\ implemented\\ on\\ CPU/CUDA',
            rationale='prbe41-11 hunk 1: upstream lines 4666-4666 -> result lines 4666-4667',
            max_span_lines=7,
        ),
    ),
)

PATCH_12 = FilePatch(
    path='ggml/src/ggml.c',
    description='channels-major SSM_CONV (nasone 33611a98): ggml/src/ggml.c',
    edits=(
        Edit(
            id='prbe41-12-01',
            anchor='struct\\ ggml_tensor\\ \\*\\ ggml_ssm_conv\\(\\\n\\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_context\\ \\*\\ ctx,\\\n\\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_tensor\\ \\ \\*\\ sx,\\\n\\ \\ \\ \\ \\ \\ \\ \\ struct\\ ggml_tensor\\ \\ \\*\\ c\\)\\ \\{\\\n',
            text='// The input (sx) memory layout is selected by `layout` and stored in op_params[0].\n// The output is [d_inner, n_t, n_s] in both cases.\nstatic struct ggml_tensor * ggml_ssm_conv_impl(\n        struct ggml_context * ctx,\n        struct ggml_tensor  * sx,\n        struct ggml_tensor  * c,\n        enum ggml_ssm_conv_layout layout) {\n',
            mode='replace',
            guard='//\\ The\\ input\\ \\(sx\\)\\ memory\\ layout\\ is\\ selected\\ by\\ `layout`\\ and\\ stored\\ in\\ op_params\\[0\\]\\.',
            rationale='prbe41-12 hunk 1: upstream lines 5659-5662 -> result lines 5659-5665',
            max_span_lines=6,
        ),
        Edit(
            id='prbe41-12-02',
            anchor='\\ \\ \\ \\ const\\ int64_t\\ n_t\\ \\ \\ \\ \\ =\\ sx\\->ne\\[0\\]\\ \\-\\ d_conv\\ \\+\\ 1;\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ const\\ int64_t\\ n_s\\ \\ \\ \\ \\ =\\ sx\\->ne\\[2\\];\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\ \\ \\ \\ GGML_ASSERT\\(sx\\->ne\\[0\\]\\ ==\\ d_conv\\ \\-\\ 1\\ \\+\\ n_t\\);\\\n\\ \\ \\ \\ GGML_ASSERT\\(sx\\->ne\\[1\\]\\ ==\\ d_inner\\);\\\n',
            text='    const int64_t n_s     = sx->ne[2];\n\n    // the time dimension index within sx depends on the layout\n    const int64_t time_ne = layout == GGML_SSM_CONV_LAYOUT_TIME_MAJOR ? sx->ne[0] : sx->ne[1];\n    const int64_t n_t     = time_ne - d_conv + 1; // tokens per sequence\n\n    // TODO: maybe support other strides than 1?\n    if (layout == GGML_SSM_CONV_LAYOUT_TIME_MAJOR) {\n        GGML_ASSERT(sx->ne[0] == d_conv - 1 + n_t);\n        GGML_ASSERT(sx->ne[1] == d_inner);\n    } else {\n        GGML_ASSERT(sx->ne[0] == d_inner);\n        GGML_ASSERT(sx->ne[1] == d_conv - 1 + n_t);\n    }\n',
            mode='replace',
            guard='//\\ the\\ time\\ dimension\\ index\\ within\\ sx\\ depends\\ on\\ the\\ layout',
            rationale='prbe41-12 hunk 2: upstream lines 5668-5673 -> result lines 5671-5684',
            max_span_lines=8,
        ),
        Edit(
            id='prbe41-12-03',
            anchor='\\\n\\ \\ \\ \\ struct\\ ggml_tensor\\ \\*\\ result\\ =\\ ggml_new_tensor_3d\\(ctx,\\ GGML_TYPE_F32,\\ d_inner,\\ n_t,\\ n_s\\);\\\n\\\n\\ \\ \\ \\ result\\->op\\ \\ \\ \\ \\ =\\ GGML_OP_SSM_CONV;\\\n',
            text='\n    struct ggml_tensor * result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_inner, n_t, n_s);\n\n    ggml_set_op_params_i32(result, 0, (int32_t) layout);\n\n    result->op     = GGML_OP_SSM_CONV;\n',
            mode='replace',
            guard='ggml_set_op_params_i32\\(result,\\ 0,\\ \\(int32_t\\)\\ layout\\);',
            rationale='prbe41-12 hunk 3: upstream lines 5677-5676 -> result lines 5688-5689',
            max_span_lines=6,
        ),
        Edit(
            id='prbe41-12-04',
            anchor='\\ \\ \\ \\ result\\->src\\[0\\]\\ =\\ sx;\\\n\\ \\ \\ \\ result\\->src\\[1\\]\\ =\\ c;\\\n\\\n\\ \\ \\ \\ return\\ result;\\\n\\}\\\n\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\\n\\\n',
            text='    result->src[0] = sx;\n    result->src[1] = c;\n\n    return result;\n}\n\nstruct ggml_tensor * ggml_ssm_conv(\n        struct ggml_context * ctx,\n        struct ggml_tensor  * sx,\n        struct ggml_tensor  * c) {\n    return ggml_ssm_conv_impl(ctx, sx, c, GGML_SSM_CONV_LAYOUT_TIME_MAJOR);\n}\n\nstruct ggml_tensor * ggml_ssm_conv_channels_major(\n        struct ggml_context * ctx,\n        struct ggml_tensor  * sx,\n        struct ggml_tensor  * c) {\n    return ggml_ssm_conv_impl(ctx, sx, c, GGML_SSM_CONV_LAYOUT_CHANNELS_MAJOR);\n}\n\nenum ggml_ssm_conv_layout ggml_ssm_conv_get_layout(const struct ggml_tensor * op) {\n    GGML_ASSERT(op->op == GGML_OP_SSM_CONV);\n    return (enum ggml_ssm_conv_layout) ggml_get_op_params_i32(op, 0);\n}\n\n// ggml_ssm_scan\n\n',
            mode='replace',
            guard='return\\ ggml_ssm_conv_impl\\(ctx,\\ sx,\\ c,\\ GGML_SSM_CONV_LAYOUT_TIME_MAJOR\\);',
            rationale='prbe41-12 hunk 4: upstream lines 5683-5682 -> result lines 5696-5714',
            max_span_lines=10,
        ),
    ),
)

PATCH_13 = FilePatch(
    path='src/models/delta-net-base.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): src/models/delta-net-base.cpp',
    edits=(
        Edit(
            id='prbe41-13-01',
            anchor='\\ \\ \\ \\ conv_states\\ =\\ ggml_reshape_3d\\(ctx0,\\ conv_states,\\ conv_kernel_size\\ \\-\\ 1,\\ conv_channels,\\ n_seqs\\);\\\n\\ \\ \\ \\ cb\\(conv_states,\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ,\\ il\\);\\\n\\\n\\ \\ \\ \\ qkv_mixed\\ =\\ ggml_transpose\\(ctx0,\\ qkv_mixed\\);\\\n\\ \\ \\ \\ cb\\(qkv_mixed,\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ,\\ il\\);\\\n\\\n\\ \\ \\ \\ ggml_tensor\\ \\*\\ conv_input\\ =\\ ggml_concat\\(ctx0,\\ conv_states,\\ qkv_mixed,\\ 0\\);\\\n',
            text='    // channels-major layout: [conv_channels, time, n_seqs] (channels contiguous).\n    // qkv_mixed already arrives channels-major as [conv_channels, n_tokens, n_seqs],\n    // so no transpose is needed: prepend the recurrent conv state along the time\n    // axis (dim 1) and feed ggml_ssm_conv_channels_major directly (nasone 33611a98).\n    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_channels, conv_kernel_size - 1, n_seqs);\n    cb(conv_states, "conv_states_reshaped", il);\n\n    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 1);\n',
            mode='replace',
            guard='//\\ channels\\-major\\ layout:\\ \\[conv_channels,\\ time,\\ n_seqs\\]\\ \\(channels\\ contiguous\\)\\.',
            rationale='prbe41-13 hunk 1: upstream lines 466-472 -> result lines 466-473',
            max_span_lines=9,
        ),
        Edit(
            id='prbe41-13-02',
            anchor='\\ \\ \\ \\ if\\ \\(cparams\\.n_rs_seq\\ ==\\ 0\\)\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ const\\ int64_t\\ s_idx\\ \\ =\\ conv_input\\->ne\\[0\\]\\ \\-\\ conv_states\\->ne\\[0\\];\\\n',
            text='    // number of time steps in conv_input (state + new tokens)\n    const int64_t n_time = conv_input->ne[1];\n\n    if (cparams.n_rs_seq == 0) {\n        const int64_t s_idx  = n_time - (conv_kernel_size - 1);\n',
            mode='replace',
            guard='//\\ number\\ of\\ time\\ steps\\ in\\ conv_input\\ \\(state\\ \\+\\ new\\ tokens\\)',
            rationale='prbe41-13 hunk 2: upstream lines 479-480 -> result lines 480-484',
            max_span_lines=4,
        ),
        Edit(
            id='prbe41-13-03',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_kernel_size\\ \\-\\ 1,\\ conv_channels,\\ n_seqs,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_input\\->nb\\[1\\],\\ conv_input\\->nb\\[2\\],\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ggml_row_size\\(conv_input\\->type,\\ s_idx\\)\\);\\\n',
            text='                    conv_channels, conv_kernel_size - 1, n_seqs,\n                    conv_input->nb[1], conv_input->nb[2],\n                    s_idx * conv_input->nb[1]);\n',
            mode='replace',
            guard='conv_channels,\\ conv_kernel_size\\ \\-\\ 1,\\ n_seqs,',
            rationale='prbe41-13 hunk 3: upstream lines 485-487 -> result lines 489-491',
            max_span_lines=5,
        ),
        Edit(
            id='prbe41-13-04',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ const\\ int64_t\\ s_idx\\ \\ =\\ std::max<int64_t>\\(0,\\ conv_input\\->ne\\[0\\]\\ \\-\\ conv_states\\->ne\\[0\\]\\ \\-\\ K\\ \\+\\ t\\);\\\n',
            text='            const int64_t s_idx  = std::max<int64_t>(0, n_time - (conv_kernel_size - 1) - K + t);\n',
            mode='replace',
            guard='const\\ int64_t\\ s_idx\\ \\ =\\ std::max<int64_t>\\(0,\\ n_time\\ \\-\\ \\(conv_kernel_size\\ \\-\\ 1\\)\\ \\-\\ K\\ \\+\\ t\\);',
            rationale='prbe41-13 hunk 4: upstream lines 505-505 -> result lines 509-509',
            max_span_lines=3,
        ),
        Edit(
            id='prbe41-13-05',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_kernel_size\\ \\-\\ 1,\\ conv_channels,\\ n_seqs,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_input\\->nb\\[1\\],\\ conv_input\\->nb\\[2\\],\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ ggml_row_size\\(conv_input\\->type,\\ s_idx\\)\\);\\\n',
            text='                        conv_channels, conv_kernel_size - 1, n_seqs,\n                        conv_input->nb[1], conv_input->nb[2],\n                        s_idx * conv_input->nb[1]);\n',
            mode='replace',
            guard='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_channels,\\ conv_kernel_size\\ \\-\\ 1,\\ n_seqs,\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ conv_input\\->nb\\[1\\],\\ conv_input\\->nb\\[2\\],\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ s_idx\\ \\*\\ conv_input\\->nb\\[1\\]\\);\\\n',
            rationale='prbe41-13 hunk 5: upstream lines 510-512 -> result lines 514-516',
            max_span_lines=5,
        ),
    ),
)

PATCH_14 = FilePatch(
    path='src/models/qwen35.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): src/models/qwen35.cpp',
    edits=(
        Edit(
            id='prbe41-14-01',
            anchor='\\ \\ \\ \\ ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv\\(ctx0,\\ conv_input,\\ conv_kernel\\);\\\n',
            text='    ggml_tensor * conv_output_proper = ggml_ssm_conv_channels_major(ctx0, conv_input, conv_kernel);\n',
            mode='replace',
            guard='ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv_channels_major\\(ctx0,\\ conv_input,\\ conv_kernel\\);',
            rationale='prbe41-14 hunk 1: upstream lines 391-391 -> result lines 391-391',
            max_span_lines=3,
        ),
    ),
)

PATCH_15 = FilePatch(
    path='src/models/qwen35moe.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): src/models/qwen35moe.cpp',
    edits=(
        Edit(
            id='prbe41-15-01',
            anchor='\\ \\ \\ \\ ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv\\(ctx0,\\ conv_input,\\ conv_kernel\\);\\\n',
            text='    ggml_tensor * conv_output_proper = ggml_ssm_conv_channels_major(ctx0, conv_input, conv_kernel);\n',
            mode='replace',
            guard='ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv_channels_major\\(ctx0,\\ conv_input,\\ conv_kernel\\);',
            rationale='prbe41-15 hunk 1: upstream lines 415-415 -> result lines 415-415',
            max_span_lines=3,
        ),
    ),
)

PATCH_16 = FilePatch(
    path='src/models/qwen3next.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): src/models/qwen3next.cpp',
    edits=(
        Edit(
            id='prbe41-16-01',
            anchor='\\ \\ \\ \\ ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv\\(ctx0,\\ conv_input,\\ conv_kernel\\);\\\n',
            text='    ggml_tensor * conv_output_proper = ggml_ssm_conv_channels_major(ctx0, conv_input, conv_kernel);\n',
            mode='replace',
            guard='ggml_tensor\\ \\*\\ conv_output_proper\\ =\\ ggml_ssm_conv_channels_major\\(ctx0,\\ conv_input,\\ conv_kernel\\);',
            rationale='prbe41-16 hunk 1: upstream lines 471-471 -> result lines 471-471',
            max_span_lines=3,
        ),
    ),
)

PATCH_17 = FilePatch(
    path='tests/test-backend-ops.cpp',
    description='channels-major SSM_CONV (nasone 33611a98): tests/test-backend-ops.cpp',
    edits=(
        Edit(
            id='prbe41-17-01',
            anchor='\\\n\\ \\ \\ \\ std::string\\ vars\\(\\)\\ override\\ \\{\\\n\\ \\ \\ \\ \\ \\ \\ \\ return\\ VARS_TO_STR3\\(type,\\ ne_a,\\ ne_b\\);\\\n',
            text='    const bool channels_major;  // if true, ne_a is [d_inner, d_conv-1+n_t, n_s, 1]\n\n    std::string vars() override {\n        return VARS_TO_STR4(type, ne_a, ne_b, channels_major);\n',
            mode='replace',
            guard='const\\ bool\\ channels_major;\\ \\ //\\ if\\ true,\\ ne_a\\ is\\ \\[d_inner,\\ d_conv\\-1\\+n_t,\\ n_s,\\ 1\\]',
            rationale='prbe41-17 hunk 1: upstream lines 4378-4380 -> result lines 4378-4381',
            max_span_lines=5,
        ),
        Edit(
            id='prbe41-17-02',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ std::array<int64_t,\\ 4>\\ ne_b\\ =\\ \\{3,\\ 3,\\ 1,\\ 1\\}\\)\\\n\\ \\ \\ \\ \\ \\ \\ \\ :\\ type\\(type\\),\\ ne_a\\(ne_a\\),\\ ne_b\\(ne_b\\)\\ \\{\\}\\\n',
            text='            std::array<int64_t, 4> ne_b = {3, 3, 1, 1},\n            bool channels_major = false)\n        : type(type), ne_a(ne_a), ne_b(ne_b), channels_major(channels_major) {}\n',
            mode='replace',
            guard='std::array<int64_t,\\ 4>\\ ne_b\\ =\\ \\{3,\\ 3,\\ 1,\\ 1\\},',
            rationale='prbe41-17 hunk 2: upstream lines 4385-4386 -> result lines 4386-4388',
            max_span_lines=4,
        ),
        Edit(
            id='prbe41-17-03',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ ggml_tensor\\ \\*\\ a\\ \\ \\ =\\ ggml_new_tensor\\(ctx,\\ type,\\ 4,\\ ne_a\\.data\\(\\)\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ ggml_tensor\\ \\*\\ b\\ \\ \\ =\\ ggml_new_tensor\\(ctx,\\ type,\\ 4,\\ ne_b\\.data\\(\\)\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ ggml_tensor\\ \\*\\ out\\ =\\ ggml_ssm_conv\\(ctx,\\ a,\\ b\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ return\\ out;\\\n\\ \\ \\ \\ \\}\\\n',
            text='        ggml_tensor * a   = ggml_new_tensor(ctx, type, 4, ne_a.data());\n        ggml_tensor * b   = ggml_new_tensor(ctx, type, 4, ne_b.data());\n        ggml_tensor * out = channels_major ? ggml_ssm_conv_channels_major(ctx, a, b)\n                                           : ggml_ssm_conv(ctx, a, b);\n        return out;\n    }\n',
            mode='replace',
            guard='ggml_tensor\\ \\*\\ out\\ =\\ channels_major\\ \\?\\ ggml_ssm_conv_channels_major\\(ctx,\\ a,\\ b\\)',
            rationale='prbe41-17 hunk 3: upstream lines 4391-4391 -> result lines 4393-4394',
            max_span_lines=7,
        ),
        Edit(
            id='prbe41-17-04',
            anchor='\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ test_cases\\.emplace_back\\(new\\ test_ssm_conv\\(GGML_TYPE_F32,\\ \\{d_conv\\ \\-\\ 1\\ \\+\\ 64,\\ d_inner,\\ 1,\\ 1\\},\\ \\{d_conv,\\ d_inner,\\ 1,\\ 1\\}\\)\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ \\ test_cases\\.emplace_back\\(new\\ test_ssm_conv\\(GGML_TYPE_F32,\\ \\{d_conv\\ \\-\\ 1\\ \\+\\ 64,\\ d_inner,\\ 4,\\ 1\\},\\ \\{d_conv,\\ d_inner,\\ 1,\\ 1\\}\\)\\);\\\n\\ \\ \\ \\ \\ \\ \\ \\ \\}\\\n\\ \\ \\ \\ \\}\\\n',
            text='            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}));\n\n            // channels-major input: ne_a is [d_inner, d_conv-1+n_t, n_s, 1]\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_inner, d_conv, 1, 1}, {d_conv, d_inner, 1, 1}, true));\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_inner, 2 * d_conv, 1, 1}, {d_conv, d_inner, 1, 1}, true));\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_inner, d_conv, 4, 1}, {d_conv, d_inner, 1, 1}, true));\n            // long token (n_t > 32, exercises the long_token kernel path)\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_inner, d_conv - 1 + 64, 1, 1}, {d_conv, d_inner, 1, 1}, true));\n            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_inner, d_conv - 1 + 64, 4, 1}, {d_conv, d_inner, 1, 1}, true));\n        }\n    }\n',
            mode='replace',
            guard='//\\ channels\\-major\\ input:\\ ne_a\\ is\\ \\[d_inner,\\ d_conv\\-1\\+n_t,\\ n_s,\\ 1\\]',
            rationale='prbe41-17 hunk 4: upstream lines 9822-9821 -> result lines 9825-9832',
            max_span_lines=6,
        ),
    ),
)

PATCHES = [PATCH_01, PATCH_02, PATCH_03, PATCH_04, PATCH_05, PATCH_06, PATCH_07, PATCH_08, PATCH_09, PATCH_10, PATCH_11, PATCH_12, PATCH_13, PATCH_14, PATCH_15, PATCH_16, PATCH_17]
