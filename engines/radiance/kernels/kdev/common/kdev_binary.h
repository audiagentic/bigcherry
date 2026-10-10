// kdev: the host side of add / mul, shared by every candidate of those ops so candidates differ only in the kernel.
// The checks and the one broadcast rule follow libr4d's own launcher (r4d_model_bf16.hip, `binary`).
#ifndef KDEV_BINARY_H
#define KDEV_BINARY_H

#include "kdev_device.h"

struct KdevBinary {
    const uint16_t* a;
    const uint16_t* b;
    uint16_t* y;
    uint32_t rows, n;
    uint32_t a_ld, b_ld, y_ld;  // row pitch in elements; b_ld == 0 means b is one row, broadcast down the rows
};

// Fill `out` from the op's operands (a, b, y), or return the status that refuses them.
static inline int kdev_binary_args(const RadArgs* args, KdevBinary* out) {
    const RadTensor* A = rad_arg_in(args, 0);
    const RadTensor* B = rad_arg_in(args, 1);
    const RadTensor* Y = rad_arg_in(args, 2);
    if (!A || !B || !Y || !A->data || !B->data || !Y->data) return RAD_E_INVAL;
    if (A->dtype != RAD_BF16 || B->dtype != RAD_BF16 || Y->dtype != RAD_BF16) return RAD_E_DTYPE;

    const long long n = rad_args_geti_or(args, "n", Y->shape[Y->rank ? Y->rank - 1 : 0]);
    if (n <= 0) return RAD_E_SHAPE;
    long long rows = 0, rows_a = 0;
    const long long y_ld = kdev_rows_pitch(Y, n, &rows);
    const long long a_ld = kdev_rows_pitch(A, n, &rows_a);
    if (y_ld < 0 || a_ld < 0) return RAD_E_STRIDE;
    if (rows <= 0 || rows_a < rows) return RAD_E_SHAPE;

    long long b_ld = 0;
    if (rad_tensor_numel(B) == n && rows > 1) {
        if (!rad_tensor_is_contiguous(B)) return RAD_E_STRIDE;
    } else {
        long long rows_b = 0;
        b_ld = kdev_rows_pitch(B, n, &rows_b);
        if (b_ld <= 0) return RAD_E_STRIDE;
        if (rows_b < rows) return RAD_E_SHAPE;
    }
    if (rows > 0x7fffffffLL / n || y_ld > 0x7fffffffLL || a_ld > 0x7fffffffLL || b_ld > 0x7fffffffLL) return RAD_E_SHAPE;

    out->a = static_cast<const uint16_t*>(A->data);
    out->b = static_cast<const uint16_t*>(B->data);
    out->y = static_cast<uint16_t*>(Y->data);
    out->rows = static_cast<uint32_t>(rows);
    out->n = static_cast<uint32_t>(n);
    out->a_ld = static_cast<uint32_t>(a_ld);
    out->b_ld = static_cast<uint32_t>(b_ld);
    out->y_ld = static_cast<uint32_t>(y_ld);
    return RAD_OK;
}

#endif
