// kdev: what every candidate plugin's row table needs, so a candidate is its rows and its kernels and nothing else.
//
//   #include "kdev_plugin.h"
//   extern "C" int my_launch(const RadArgs*, RadStream);
//   static const RadConstraint cBf16[] = { RAD_CIN("dtype", "bf16") };
//   static const RadKernelInfo rows[] = {
//       KDEV_ROW("add_mine", "add", "elem", "what it computes", cBf16, my_launch, kdev_shape_binary),
//   };
//   KDEV_PLUGIN(rows)
//
// The plugin's name is KDEV_PLUGIN_NAME (k_<candidate directory>, set by the build), which is the name rad-kbench
// prints and --kernels takes. Schemas are not declared here: a candidate implements ops libref already defines.
#ifndef KDEV_PLUGIN_H
#define KDEV_PLUGIN_H

#include "rad_abi.h"
#include "rad_plugin.h"

#include <cstring>

#ifndef KDEV_PLUGIN_NAME
#error "KDEV_PLUGIN_NAME is set by engines/radiance/kernels/kdev/CMakeLists.txt"
#endif
#ifndef KDEV_TARGETS
#define KDEV_TARGETS ""
#endif

#define KDEV_COUNT(a) (static_cast<int>(sizeof(a) / sizeof((a)[0])))

// One device row with nothing to tune. `shape_fn` describes the operands so rad-kbench can call the kernel cold;
// without it the tool skips the row by name.
#define KDEV_ROW(kernel_name, op_name, family_name, what, constraint_array, launch_fn, shape_fn) \
    RadKernelInfo{                                                                               \
        .name = kernel_name, .op = op_name, .family = family_name, .computes = what,            \
        .shape = "see the candidate's source", .dtypes = "see constraints",                     \
        .domain = RAD_DOMAIN_DEVICE, .priority = 10,                                             \
        .constraints = constraint_array, .n_constraints = KDEV_COUNT(constraint_array),         \
        .launch = launch_fn, .opd_shape = shape_fn,                                              \
    }

#define KDEV_PLUGIN(rows)                                                                         \
    static const RadPluginInfo kdev_info = {                                                      \
        RAD_PLUGIN_KERNEL, KDEV_PLUGIN_NAME, "0.0.0",                                             \
        "kdev candidate: a kernel under development, not a serving library", KDEV_TARGETS,       \
    };                                                                                            \
    extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }                  \
    extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kdev_info; }                  \
    extern "C" int rad_kernel_schema_count(void) { return 0; }                                    \
    extern "C" const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }                   \
    extern "C" int rad_kernel_count(void) { return KDEV_COUNT(rows); }                            \
    extern "C" const RadKernelInfo* rad_kernel_at(int i) {                                        \
        return i >= 0 && i < KDEV_COUNT(rows) ? &(rows)[i] : nullptr;                             \
    }                                                                                             \
    extern "C" int rad_kernel_concurrent(int i) { return i >= 0 && i < KDEV_COUNT(rows); }

// ---------------------------------------------------------------- operand descriptions
// The same geometry libref describes for the op, so rad-kbench compares like with like.

static inline uint32_t kdev_act_dtype(const RadParam* p, int n_p) {
    const char* name = rad_param_gets(p, n_p, "dtype", "bf16");
    return rad_dtype_parse(name);
}

static inline int kdev_dense(RadOpdDesc* out, uint32_t dtype, long long rows, long long cols) {
    if (!out || rows < 1 || cols < 1 || dtype == RAD_DT_INVALID) return RAD_E_SHAPE;
    *out = {};
    out->dtype = dtype;
    out->rank = 2;
    out->shape[0] = rows;
    out->shape[1] = cols;
    out->fill = RAD_FILL_NORMAL;
    out->idx_const = -1;
    return RAD_OK;
}

// add, mul: a [M, n], b [M, n], y [M, n]
static inline int kdev_shape_binary(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (operand < 0 || operand > 2) return RAD_E_SHAPE;
    return kdev_dense(out, kdev_act_dtype(p, n_p), rad_param_getdim(p, n_p, "M", 0),
                      rad_param_getdim(p, n_p, "n", 0));
}

#endif
