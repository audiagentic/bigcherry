"""PNRO10: Add ctx_other model devices to speculative scheduler backends.

When a speculative context (MTP/draft) has a ctx_other (the target context),
shared tensors between the two contexts must be schedulable on valid backends.
The target context's model devices may include backends that the draft context
does not have (e.g., if the target uses a different GPU partition).

This patch adds the ctx_other model devices to the draft context's backend list,
deduplicating by backend device handle and preserving the existing ordering
relative to ACCEL/CPU backends.
"""

from bigcherry.patcher import Edit, FilePatch

PATCHES = [
    FilePatch(
        path="vendor/llama.cpp/src/llama-context.cpp",
        edits=(
            Edit(
                id="pnro10_add_ctx_other_devices",
                anchor=r"backends\.emplace_back\(backend\);",
                text=(
                    "\n        // PNRO10: add ctx_other model devices to the scheduler backends\n"
                    "        // so that shared tensors between the draft and target contexts\n"
                    "        // can be scheduled on valid backends.\n"
                    "        if (cparams.ctx_other != nullptr) {\n"
                    "            llama_context * other = cparams.ctx_other;\n"
                    "            llama_model * other_model = llama_get_model(other);\n"
                    "            if (other_model != nullptr) {\n"
                    "                for (const auto & dev : other_model->devices) {\n"
                    "                    // deduplicate by backend device handle\n"
                    "                    bool already_present = false;\n"
                    "                    for (auto & existing : backends) {\n"
                    "                        if (ggml_backend_get_device(existing.get()) == dev.dev) {\n"
                    "                            already_present = true;\n"
                    "                            break;\n"
                    "                        }\n"
                    "                    }\n"
                    "                    if (!already_present) {\n"
                    "                        ggml_backend_t other_backend = ggml_backend_dev_init(dev.dev, nullptr);\n"
                    "                        if (other_backend == nullptr) {\n"
                    "                            throw std::runtime_error(\n"
                    "                                format(\"failed to initialize %s backend from ctx_other\",\n"
                    "                                       ggml_backend_dev_name(dev.dev)));\n"
                    "                        }\n"
                    "                        backends.emplace_back(other_backend);\n"
                    "                    }\n"
                    "                }\n"
                    "            }\n"
                    "        }"
                ),
                mode="insert_after",
                guard="PNRO10: add ctx_other model devices",
                rationale="Add ctx_other model devices to the scheduler backends so that shared tensors between the draft and target contexts can be scheduled on valid backends.",
                occurrence=0,
            ),
        ),
    ),
]
