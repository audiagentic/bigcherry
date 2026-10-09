#include "kdev_plugin.h"

extern "C" int k_rows_add(const RadArgs*, RadStream);
extern "C" int k_rows_mul(const RadArgs*, RadStream);

static const RadConstraint cBf16[] = { RAD_CIN("dtype", "bf16") };
static const RadKernelInfo rows[] = {
    KDEV_ROW("rows_add", "add", "elem", "bf16 add, a run of 8 elements per thread", cBf16, k_rows_add, kdev_shape_binary),
    KDEV_ROW("rows_mul", "mul", "elem", "bf16 mul, a run of 8 elements per thread", cBf16, k_rows_mul, kdev_shape_binary),
};
KDEV_PLUGIN(rows)
