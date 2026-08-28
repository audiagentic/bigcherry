"""HI134 - attribute META reduction copy activity to its transfer stages.

The existing HI58/0830 telemetry records one observation per reduction and
already bridges generic META to the HIP owner through proc-address callbacks.
This diagnostic adds only a bounded stage trace: the generic fallback reports
the two copy sites it owns, while the HIP telemetry sink stores and serializes
the records in that same observation event.
"""

GROUP = "core"
STATE = "untested"
REQUIRES = ("0830_split_reduce_telemetry",)

from bigcherry.patcher import Edit, FilePatch


CUDA = FilePatch(
    path="ggml/src/ggml-cuda/ggml-cuda.cu",
    description="bridge generic META transfer stages into HIP reduction telemetry",
    edits=(
        Edit(
            id="meta-stage-telemetry-wrapper",
            anchor=r'^static void \* ggml_backend_cuda_reg_get_proc_address\(ggml_backend_reg_t reg, const char \* name\) \{$',
            rationale="the existing CUDA proc-address bridge used by HI58",
            mode="insert_before",
            text=(
                '#ifdef GGML_HIP_DISPATCH\n'
                'static void ggml_backend_cuda_comm_telemetry_stage(\n'
                '        uint16_t phase, uint16_t step, int16_t src_rank, int16_t dst_rank,\n'
                '        uint64_t bytes, const ggml_tensor * source) {\n'
                '    ggml_hip_reduce_telemetry_meta_stage(\n'
                '        phase, step, src_rank, dst_rank, bytes, source);\n'
                '}\n'
                '#endif\n\n'
            ),
            guard=r'ggml_backend_cuda_comm_telemetry_stage\(',
        ),
        Edit(
            id="meta-stage-telemetry-proc",
            anchor=(r'^        return \(void \*\)ggml_backend_cuda_comm_telemetry_fallback;\n'
                    r'^    \}$'),
            rationale="publish the optional stage sink beside the existing fallback observer",
            mode="insert_after",
            text=(
                '\n#ifdef GGML_HIP_DISPATCH\n'
                '    if (strcmp(name, "ggml_backend_comm_telemetry_stage") == 0) {\n'
                '        return (void *)ggml_backend_cuda_comm_telemetry_stage;\n'
                '    }\n'
                '#endif\n'
            ),
            guard=r'ggml_backend_comm_telemetry_stage"',
        ),
    ),
)


META = FilePatch(
    path="ggml/src/ggml-backend-meta.cpp",
    description="record META fold, butterfly and copy-back transfer stages",
    edits=(
        Edit(
            id="meta-stage-telemetry-type",
            anchor=r'^using ggml_backend_comm_telemetry_fallback_t = void \(\*\)\(void \*, ggml_tensor \*\*, const char \*, size_t\);$',
            rationale="keep the optional stage callback beside the existing HI58 callback type",
            mode="insert_before",
            text=(
                'enum {\n'
                '    GGML_META_STAGE_PHASE_FOLD = 1,\n'
                '    GGML_META_STAGE_PHASE_BUTTERFLY = 2,\n'
                '    GGML_META_STAGE_PHASE_COPY_BACK = 3,\n'
                '};\n'
                'using ggml_backend_comm_telemetry_stage_t = void (*)(\n'
                '        uint16_t, uint16_t, int16_t, int16_t, uint64_t, const ggml_tensor *);\n\n'
            ),
            guard=r'ggml_backend_comm_telemetry_stage_t',
        ),
        Edit(
            id="meta-stage-telemetry-field",
            anchor=r'^    ggml_backend_comm_telemetry_fallback_t comm_fallback = nullptr;$',
            rationale="retain the optional stage callback in the generic META context",
            mode="insert_after",
            text='\n    ggml_backend_comm_telemetry_stage_t comm_stage = nullptr;',
            guard=r'comm_stage = nullptr',
        ),
        Edit(
            id="meta-stage-telemetry-resolve",
            anchor=(r'^            comm_fallback = \(ggml_backend_comm_telemetry_fallback_t\)\n'
                    r'^                ggml_backend_reg_get_proc_address\(ggml_backend_dev_backend_reg\(\n'
                    r'^                    ggml_backend_get_device\(simple_backends\[0\]\)\),\n'
                    r'^                    \s*\);$'),
            rationale="resolve the optional stage sink through the same backend registry",
            mode="insert_after",
            text=(
                '\n            comm_stage = (ggml_backend_comm_telemetry_stage_t)\n'
                '                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(\n'
                '                    ggml_backend_get_device(simple_backends[0])),\n'
                '                    "ggml_backend_comm_telemetry_stage");'
            ),
            guard=r'comm_stage = \(ggml_backend_comm_telemetry_stage_t\)',
        ),
        Edit(
            id="meta-stage-telemetry-push-signature",
            anchor=r'^        auto push_data = \[&\]\(const size_t j_src, const size_t j_dst, const size_t i_buf\) \{$',
            rationale="carry stage identity from the fold/butterfly loop into the shared copy site",
            mode="replace",
            text=(
                '        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf,\n'
                '                             const uint16_t phase, const uint16_t step) {'
            ),
            guard=r'const uint16_t phase, const uint16_t step',
        ),
        Edit(
            id="meta-stage-telemetry-push-call",
            anchor=r'^            ggml_backend_tensor_copy_async\(bcj_src\.backend, bcj_dst\.backend, node_src, node_tmp\);$',
            rationale="observe the shared fold/butterfly copy before submission without changing it",
            mode="insert_before",
            occurrence=0,
            expect_matches=1,
            text=(
                '            if (backend_ctx->comm_stage != nullptr) {\n'
                '                backend_ctx->comm_stage(phase, step,\n'
                '                    static_cast<int16_t>(j_src), static_cast<int16_t>(j_dst),\n'
                '                    static_cast<uint64_t>(ggml_nbytes(node_src)), node_src);\n'
                '            }\n'
            ),
            guard=r'backend_ctx->comm_stage\(phase, step',
        ),
        Edit(
            id="meta-stage-telemetry-fold",
            anchor=r'^            push_data\(j_src, j_dst, i_buf\);$',
            rationale="label the non-power-of-two excess-rank fold",
            mode="replace",
            text=(
                '            push_data(j_src, j_dst, i_buf, GGML_META_STAGE_PHASE_FOLD,\n'
                '                      static_cast<uint16_t>(i_buf));'
            ),
            guard=r'push_data\(j_src, j_dst, i_buf, GGML_META_STAGE_PHASE_FOLD',
        ),
        Edit(
            id="meta-stage-telemetry-butterfly",
            anchor=r'^                push_data\(j, j_other, i_buf\);$',
            rationale="label each butterfly transfer by its butterfly step",
            mode="replace",
            text=(
                '                push_data(j, j_other, i_buf, GGML_META_STAGE_PHASE_BUTTERFLY,\n'
                '                          static_cast<uint16_t>(offset_j));'
            ),
            guard=r'push_data\(j, j_other, i_buf, GGML_META_STAGE_PHASE_BUTTERFLY',
        ),
        Edit(
            id="meta-stage-telemetry-copy-back",
            anchor=r'^            ggml_backend_tensor_copy_async\(bcj_src\.backend, bcj_dst\.backend, node_src, node_dst\);$',
            rationale="observe the final copy to each non-power-of-two excess rank",
            mode="insert_before",
            text=(
                '            if (backend_ctx->comm_stage != nullptr) {\n'
                '                backend_ctx->comm_stage(GGML_META_STAGE_PHASE_COPY_BACK,\n'
                '                    static_cast<uint16_t>(i_buf),\n'
                '                    static_cast<int16_t>(j - 2*offset_j_max),\n'
                '                    static_cast<int16_t>(j),\n'
                '                    static_cast<uint64_t>(ggml_nbytes(node_src)), node_src);\n'
                '            }\n'
            ),
            guard=r'comm_stage\(GGML_META_STAGE_PHASE_COPY_BACK',
        ),
    ),
)


PATCHES = [CUDA, META]
