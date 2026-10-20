// bc_hc_read.hip -- hc_read for decode-sized steps on gfx1201 (RAD15). Stage 1: declines every call, so the row hands each
// call to libr4d's kernel; this proves the plugin's selection, schema and layout plumbing before the kernel is replaced.
#include "rad_abi.h"

extern "C" int bc_hc_read_decode(const RadArgs* a, RadStream s) {
    (void)a;
    (void)s;
    return RAD_E_UNSUPPORTED;
}
