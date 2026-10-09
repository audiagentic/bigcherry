// A deliberately incomplete gfx1100 Radiance plugin. Never advertise an untested band.
#include "rad_abi.h"
#include <cstring>

extern "C" int r11_add_bf16(const RadArgs*, RadStream);
extern "C" int r11_mul_bf16(const RadArgs*, RadStream);

static const RadConstraint kBf16[] = { RAD_CIN("dtype", "bf16") };
static const RadKernelInfo kKernels[] = {
    {
        .name = "r11_add_bf16",
        .op = "add",
        .family = "elem",
        .computes = "BF16 elementwise add; b may broadcast a single row",
        .shape = "M >= 1, n >= 1; dense/contiguous tensors",
        .dtypes = "bf16",
        .domain = RAD_DOMAIN_DEVICE,
        .priority = 10,
        .constraints = kBf16,
        .n_constraints = 1,
        .launch = r11_add_bf16,
    },
    {
        .name = "r11_mul_bf16",
        .op = "mul",
        .family = "elem",
        .computes = "BF16 elementwise mul; b may broadcast a single row",
        .shape = "M >= 1, n >= 1; dense/contiguous tensors",
        .dtypes = "bf16",
        .domain = RAD_DOMAIN_DEVICE,
        .priority = 10,
        .constraints = kBf16,
        .n_constraints = 1,
        .launch = r11_mul_bf16,
    },
};
static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL, "libr11", "0.0.1",
    "Experimental gfx1100 kernels; NOT a complete standalone Radiance backend",
    "gfx1100",
};
extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }
// Schemas for conventional add/mul are supplied by Radiance libref.
extern "C" int rad_kernel_schema_count(void) { return 0; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
extern "C" int rad_kernel_count(void) {
    return static_cast<int>(sizeof(kKernels) / sizeof(kKernels[0]));
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < rad_kernel_count() ? &kKernels[i] : nullptr;
}
extern "C" int rad_kernel_concurrent(int i) {
    return i >= 0 && i < rad_kernel_count();
}
