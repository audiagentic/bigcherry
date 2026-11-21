// Verify plugin kernel enumeration follows each bound helper generation, not
// the state seen on the first loader query.
#include "rad_abi.h"
#include <cassert>

extern "C" int rad_kernel_count(void);
extern "C" const RadKernelInfo* rad_kernel_at(int);

namespace {
bool enabled = false;
RadKernelInfo rows[2]{};
}

extern "C" int r3_moe_mailbox_enabled(void) { return enabled ? 1 : 0; }
extern "C" int r3_moe_mailbox_schema_count(void) { return 0; }
extern "C" const RadOpSchema* r3_moe_mailbox_schema_at(int) { return nullptr; }
extern "C" const RadKernelInfo* r3_moe_mailbox_row(int index) {
    return index >= 0 && index < 2 ? &rows[index] : nullptr;
}

int main() {
    assert(rad_kernel_count() == 2);
    enabled = true;
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
    assert(rad_kernel_at(1) == &rows[1]);
    assert(rad_kernel_at(2) == nullptr);

    enabled = false;
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
    enabled = true;
    assert(rad_kernel_count() == 2);
    assert(rad_kernel_at(0) == &rows[0]);
}
