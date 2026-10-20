// libr3ngram kernel table: the opt-in n-gram embedding gather (R3_NGRAM_DEV), a HOST row whose sparse gather runs on a
// non-rank GPU through helpers/libr3_ngram_helper.so. A plugin of its own (RCS06), split out of libr3: it has no tie to
// the gfx11 kernel overrides or the MoE path. Its op's schema is declared by the engine's kernel library (libr4d or libr3),
// so this plugin declares none. Shelved (evaluated, default off; the R9700 is compute, owner 2026-10-13).
#include "rad_abi.h"

extern "C" {
int r3_ngram_enabled(void);
const RadKernelInfo* r3_ngram_row(void);
}

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = {
        RAD_PLUGIN_KERNEL, "libr3ngram", "0.0.1",
        "opt-in n-gram embedding gather on a spare GPU (R3_NGRAM_DEV)", R3NGRAM_TARGETS,
    };
    return &info;
}
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) { return r3_ngram_enabled() ? 1 : 0; }
extern "C" const RadKernelInfo* rad_kernel_at(int i) { return i == 0 && r3_ngram_enabled() ? r3_ngram_row() : nullptr; }
extern "C" int rad_kernel_concurrent(int) { return 0; }
