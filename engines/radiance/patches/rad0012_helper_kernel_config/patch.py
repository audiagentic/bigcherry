"""RSC06: explicit per-process helper settings for libr4d and its gfx11/gfx10 ports."""
from bigcherry.patch.apply import Edit, FilePatch

HELPER_CONFIG_H = r"""/* BigCherry helper configuration. Install once, before any plugin callback or worker thread.
 * An unconfigured library belongs to the engine; its environment reads remain for RSC07.
 * There is no environment-derived helper configuration. All fields are fixed-width ABI values. */
#ifndef BC_R4D_HELPER_CONFIG_H
#define BC_R4D_HELPER_CONFIG_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct BcR4dHelperConfig {
    uint32_t struct_size;
    uint32_t native, fp16, ahalf;
    uint32_t moe_probe; /* 0 kernel, 1 no WMMA, 2 no widening; experiment only */
    uint32_t refusal_trace;
} BcR4dHelperConfig;
/* Returns 0 on success, -1 on invalid size/value or repeated installation. */
int bc_r4d_configure_helper(const BcR4dHelperConfig* config);
const BcR4dHelperConfig* bc_r4d_helper_config(void);
#ifdef __cplusplus
}
#endif

#ifdef BC_R4D_HELPER_CONFIG_IMPLEMENTATION
static BcR4dHelperConfig bc_helper_config;
int bc_r4d_configure_helper(const BcR4dHelperConfig* c) {
    if (!c || c->struct_size != sizeof(BcR4dHelperConfig) || bc_helper_config.struct_size ||
        c->native > 1 || c->fp16 > 1 || c->ahalf > 1 || c->moe_probe > 2 || c->refusal_trace > 1)
        return -1;
    bc_helper_config = *c;
    return 0;
}
const BcR4dHelperConfig* bc_r4d_helper_config(void) {
    return bc_helper_config.struct_size ? &bc_helper_config : 0;
}
#endif
#endif
"""

PATCHES = (
    FilePatch(
        path="libr4d/bc_helper_config.h", create=True,
        description="Create the size-checked immutable helper configuration ABI.",
        edits=(Edit(id="create-helper-config", anchor=r"\A", text=HELPER_CONFIG_H,
                    guard=r"^#define BC_R4D_HELPER_CONFIG_H$", expect_matches=1,
                    rationale="A created file must have no upstream collision."),),
    ),
    FilePatch(
        path="libr4d/r4d_rows.cpp",
        description="Own one configuration instance in each loaded kernel library.",
        edits=(Edit(id="own-helper-config", anchor=r'^#include "r4d_plugin\.h"$',
                    mode="insert_before",
                    text='#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION\n#include "bc_helper_config.h"\n#undef BC_R4D_HELPER_CONFIG_IMPLEMENTATION\n',
                    guard=r"^#define BC_R4D_HELPER_CONFIG_IMPLEMENTATION$", expect_matches=1,
                    rationale="The row table is compiled once by libr4d, libr3 and libr2."),),
    ),
    FilePatch(
        path="libr4d/r4d_args.h",
        description="Helper refusal tracing uses typed settings; engine keeps its current read.",
        edits=(
            Edit(id="include-helper-config", anchor=r'^#include "rad_plugin\.h"$',
                 text='\n#include "bc_helper_config.h"',
                 guard=r'^#include "bc_helper_config\.h"$', expect_matches=1,
                 rationale="The shared argument helpers are host-only and used by every MoE callback."),
            Edit(id="typed-helper-trace", anchor=r"^static inline int r4d_trace_on\(void\) \{$",
                 text="\n    if (const BcR4dHelperConfig* c = bc_r4d_helper_config()) return (int)c->refusal_trace;",
                 guard=r"bc_r4d_helper_config\(\)\) return \(int\)c->refusal_trace;", expect_matches=1,
                 rationale="Branch before the engine's cached environment read, even on refused shapes."),
        ),
    ),
)
