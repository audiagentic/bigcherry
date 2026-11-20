// LT3 component harness: 6900 text worker versus same-card text and request-epoch handoff.
// Synthetic bytes: a protocol invariant, not a Gemma4 kernel or device IPC test.
#include "ltx_tp3_reference.h"
#include "ltx_tp3_protocol.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace bc_ltx_tp3_reference;
using namespace bc_ltx_tp3_protocol;
namespace {
template<class F> void refuse(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
void text_roles(bool remote) {
    const auto text = owners(Stage::gemma, remote);
    const auto connector = owners(Stage::connectors, remote);
    assert(text[remote ? 3 : 0]);
    assert(!text[remote ? 0 : 3]);
    assert((connector == std::array<bool, 4>{true, false, false, false}));
    const auto dit = owners(Stage::dit, remote);
    assert((dit == std::array<bool, 4>{true, true, true, false}));
    // 49 Gemma layers (embedding + 48) * 3840 features * 2 prompt contexts.
    assert(gemma_stack_bytes(1024, 2, 3840, 48, 2) == 770703360LL);
    // Only connector outputs (not the huge layer stack) reach the TP3 ranks.
    assert(conditioning_bytes(1024, 2, 4096, 2048, 2) == 25165824LL);
}
void ticket_cycles() {
    ConditioningMailbox input;
    for (std::uint64_t epoch = 1; epoch <= 4; ++epoch) {
        const Ticket tag{37, epoch, 2, 7, 2};
        std::vector<std::uint8_t> payload(payload_bytes(tag));
        for (std::size_t i = 0; i < payload.size(); ++i)
            payload[i] = std::uint8_t((i + epoch * 17) & 255);
        input.publish(tag, payload);
        Ticket older = tag; older.epoch--;
        refuse([&] { input.receive(older); });
        Ticket other = tag; other.request++;
        refuse([&] { input.receive(other); });
        Ticket wrong = tag; wrong.columns++;
        refuse([&] { input.receive(wrong); });
        refuse([&] { input.publish(tag, payload); });
        assert(input.receive(tag) == payload);
        refuse([&] { input.receive(tag); });
    }
    const Ticket cancellation{38, 1, 3, 11, 2};
    input.publish(cancellation, std::vector<std::uint8_t>(payload_bytes(cancellation), 10));
    input.cancel();
    refuse([&] { input.receive(cancellation); });
    const Ticket latest{38, 2, 3, 11, 2};
    input.publish(latest, std::vector<std::uint8_t>(payload_bytes(latest), 11));
    assert(input.receive(latest).size() == payload_bytes(latest));
}
void memory_admission() {
    FourCardPlan plan{{{{0, Arch::gfx1100, 24576, 512},
                        {1, Arch::gfx1100, 24576, 512},
                        {2, Arch::gfx1201, 32768, 1024}}},
                        {3, Arch::gfx1030, 16384, 1024}, 12288, 1024, 1024};
    validate_four_card_plan(plan);
    plan.text_workspace_mib = 3072;
    refuse([&] { validate_four_card_plan(plan); });
    plan.text_workspace_mib = 1024;
    plan.text.id = 2;
    refuse([&] { validate_four_card_plan(plan); });
}
} // namespace
int main() {
    text_roles(false);
    text_roles(true);
    ticket_cycles();
    memory_admission();
    std::cout << "LT3 text-worker and conditioning transfer: passed\n";
}
