#include "kdev_plugin.h"

extern "C" int k_flat_add(const RadArgs*, RadStream);
extern "C" int k_flat_mul(const RadArgs*, RadStream);

static const RadConstraint cBf16[] = { RAD_CIN("dtype", "bf16") };
static const RadKernelInfo rows[] = {
    KDEV_ROW("flat_add", "add", "elem", "bf16 add, one thread per element", cBf16, k_flat_add, kdev_shape_binary),
    KDEV_ROW("flat_mul", "mul", "elem", "bf16 mul, one thread per element", cBf16, k_flat_mul, kdev_shape_binary),
};
KDEV_PLUGIN(rows)
