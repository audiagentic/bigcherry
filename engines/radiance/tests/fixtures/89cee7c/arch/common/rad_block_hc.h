#ifndef RAD_BLOCK_HC_H
#define RAD_BLOCK_HC_H

#include "rad_fp8.h"

struct HyperConn {
    int ar_take() const { return op_ar_write6 ? kArOutBoth : kArOutExact; }
    bool takes_gather(int64_t T) const {
        return takes_ar(T) && gather_taken(g, T, gather_rows, gather_rows6);
    }
};
