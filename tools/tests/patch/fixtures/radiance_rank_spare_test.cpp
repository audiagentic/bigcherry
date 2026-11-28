// Actual vram_budget.cpp compiled with function sections, only the spare helper linked.
#include "mem/vram_budget.h"
#include <cassert>
#include <vector>
int main() {
    constexpr int64_t MiB = 1024 * 1024;
    rad::Config c{};
    c.devices = {0,1,2};
    c.gpu_headroom_mib = 512;
    std::vector<rad::VramFacts> f(3);
    f[0].free = 24000 * MiB;
    f[1].free = 24000 * MiB;
    f[2].free = 32000 * MiB;
    assert(rad::vram_budget_rank_spare_mib(c,f,2) == 0);
    assert(rad::vram_budget_rank_spare_mib(c,f,2,false) == 8000);
    c.card_headroom_mib[2] = 2048;
    assert(rad::vram_budget_rank_spare_mib(c,f,2,false) == 6464);
    assert(rad::vram_budget_rank_spare_mib(c,f,2) == 6464);
    f[2].host_reserve = 4096 * MiB;
    assert(rad::vram_budget_rank_spare_mib(c,f,2,false) == 4416);
    assert(rad::vram_budget_rank_spare_mib(c,f,0,false) == 0);
    c.vram_claim = 1;
    assert(rad::vram_budget_rank_spare_mib(c,f,2,false) == 0);
    assert(rad::vram_budget_rank_spare_mib(c,f,3,false) == 0);
}
