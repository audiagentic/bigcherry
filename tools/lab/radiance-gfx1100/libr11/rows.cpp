// A deliberately incomplete gfx1100 Radiance plugin. Never advertise an untested band.
#include "rad_abi.h"
#include "rad_plugin.h"
#include <cstring>

extern "C" int r11_add_bf16(const RadArgs*, RadStream);
extern "C" int r11_mul_bf16(const RadArgs*, RadStream);

// Describe the full-size (non-broadcast) operand geometry for rad-kbench.
// The broadcast variant is covered separately by r11_selftest.
static int binary_shape(const RadParam* p, int n_p, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= 3) return RAD_E_SHAPE;
    const long long m = rad_param_getdim(p, n_p, "M", 0);
    const long long n = rad_param_getdim(p, n_p, "n", 0);
    const char* dtype = rad_param_gets(p, n_p, "dtype", nullptr);
    if (m < 1 || n < 1 || !dtype || std::strcmp(dtype, "bf16") != 0)
        return RAD_E_SHAPE;
    *out = {};
    out->dtype = RAD_BF16;
    out->rank = 2;
    out->shape[0] = m;
    out->shape[1] = n;
    out->fill = RAD_FILL_NORMAL;
    return RAD_OK;
}

// Radiance gemm_nt: A[M,K] x B[N,K]^T -> Y[M,N] (BF16); BF16 WMMA/FP32 accumulate.
// The matrix dimensions follow upstream libr4d's shape hook, not a transposed B view.
extern "C" int r11_gemm_bf16_scalar(const RadArgs*, RadStream);
extern "C" int r11_gemm_bf16_wmma(const RadArgs*, RadStream);
static int gemm_shape(const RadParam* p, int np, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand > 2) return RAD_E_SHAPE;
    const long long m=rad_param_getdim(p,np,"M",0);
    const long long n=rad_param_getdim(p,np,"N",0);
    const long long k=rad_param_getdim(p,np,"K",0);
    const char* dtype=rad_param_gets(p,np,"dtype",nullptr);
    if (m<1 || m>16 || n<1 || n>32768 || k<16 || k>8192 ||
        k%16 || !dtype || std::strcmp(dtype,"bf16"))
        return RAD_E_SHAPE;
    *out={};
    out->dtype=RAD_BF16;
    out->rank=2;
    out->shape[0]=(operand==1?n:m);
    out->shape[1]=(operand==2?n:k);
    out->fill=RAD_FILL_NORMAL;
    out->idx_const=-1;
    return RAD_OK;
}
static const RadConstraint kBf16Gemm[] = {
    RAD_CIN("dtype","bf16"), RAD_CGE("M",1), RAD_CLE("M",16),
    RAD_CGE("N",1), RAD_CLE("N",32768),
    RAD_CGE("K",16), RAD_CLE("K",8192), RAD_CDIV("K",16)
};

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
        .opd_shape = binary_shape,
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
        .opd_shape = binary_shape,
    },
    {
        .name = "r11_gemm_bf16_scalar",
        .op = "gemm_nt",
        .family = "gemm",
        .computes = "BF16 A[M,K] x B[N,K]^T, scalar FP32 accumulate (independent GPU control)",
        .shape = "1<=M<=16, 1<=N<=32768, K multiple 16 and <=8192; contiguous",
        .dtypes = "bf16",
        .domain = RAD_DOMAIN_DEVICE,
        .priority = 2,
        .constraints = kBf16Gemm,
        .n_constraints = sizeof(kBf16Gemm)/sizeof(kBf16Gemm[0]),
        .launch = r11_gemm_bf16_scalar,
        .opd_shape = gemm_shape,
    },
    {
        .name = "r11_gemm_bf16_wmma_gfx1100",
        .op = "gemm_nt",
        .family = "gemm",
        .computes = "gfx11 wave32 BF16 WMMA, FP32 accumulate; RDNA3 interleaved output",
        .shape = "1<=M<=16, 1<=N<=32768, K multiple 16 and <=8192; contiguous",
        .dtypes = "bf16",
        .domain = RAD_DOMAIN_DEVICE,
        .priority = 10,
        .constraints = kBf16Gemm,
        .n_constraints = sizeof(kBf16Gemm)/sizeof(kBf16Gemm[0]),
        .launch = r11_gemm_bf16_wmma,
        .opd_shape = gemm_shape,
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
