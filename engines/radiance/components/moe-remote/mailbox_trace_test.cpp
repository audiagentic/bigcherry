#include "moe_remote_shm.h"
#include "mailbox_trace_math.h"
#include <cassert>
#include <cstddef>
using namespace r3::moe_shm;
using namespace r3::moe_trace;
int main() {
    static_assert(sizeof(Request) == 256);
    static_assert(offsetof(Request, mailbox_trace) == 208);
    Request r{};
    r.mailbox_trace[PublishBegin] = 1000;
    r.mailbox_trace[PublishReady] = 2500;
    r.mailbox_trace[WaitExit] = 6500;
    r.mailbox_trace[FinishBegin] = 7000;
    r.mailbox_trace[FinishEnd] = 7400;
    r.mailbox_trace[CombineWaitTicks] = 500;
    auto m = compute(r.mailbox_trace, 6000, .01);
    assert(m.valid && m.publish_us == 15 && m.wait_bookkeeping_us == 5);
    assert(m.finish_us == 4 && m.combine_engine_wait_us == 5);
    r.mailbox_trace[FinishEnd] = 0;
    assert(!compute(r.mailbox_trace, 6000, .01).valid);
    r.mailbox_trace[FinishEnd] = 7400;
    assert(!compute(r.mailbox_trace, 7500, .01).valid);
    assert(!compute(r.mailbox_trace, 6000, 0).valid);
}
