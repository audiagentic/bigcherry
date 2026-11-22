// Kernel enumeration is fixed: the rows do not depend on any helper generation (eligibility is bound per generation
// at kernel init, through the engine's tp_group).
#include "rad_abi.h"
#include <cassert>

extern "C" int rad_kernel_count(void);
extern "C" const RadKernelInfo* rad_kernel_at(int);

namespace {
RadKernelInfo rows[2]{};
}

extern "C" int r3_moe_mailbox_schema_count(void) { return 0; }
extern "C" const RadOpSchema* r3_moe_mailbox_schema_at(int) { return nullptr; }
extern "C" const RadKernelInfo* r3_moe_mailbox_row(int index) {
    return index >= 0 && index < 2 ? &rows[index] : nullptr;
}

int main() {
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
    assert(rad_kernel_at(1) == &rows[1]);
    assert(rad_kernel_at(2) == nullptr);

    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
}
