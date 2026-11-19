#include <algorithm>
#include "ltx_tp3_protocol.h"
#include <array>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace bc_ltx_tp3_protocol;

template<class F> void refuse(F f) {
    bool threw=false;
    try { f(); } catch(const std::invalid_argument&) { threw=true; }
    assert(threw);
}

void schedule_test() {
    const auto plain=collective_trace(false,false);
    const auto stg=collective_trace(true,false);
    const auto iso=collective_trace(false,true);
    const auto both=collective_trace(true,true);
    assert(plain.size()==960);
    assert(stg.size()+4==plain.size()); // Q and K of two self-attention streams skipped
    assert(iso.size()+kDitBlocks*6==plain.size()); // 2 cross-modal x (Q, K, output)
    assert(both.size()+4+kDitBlocks*6==plain.size());
    require_agreement({plain,plain,plain});
    refuse([&]{ require_agreement({plain,stg,plain}); });
    refuse([&]{ require_agreement({plain,iso,plain}); });
    refuse([&]{ collective_trace(false,false,47); });
    const auto it=std::find(stg.begin(),stg.end(),kStgBlock*100+3);
    assert(it!=stg.end()); // attention output still synchronized at STG block
    assert(std::find(stg.begin(),stg.end(),kStgBlock*100+1)==stg.end());
}

void ownership_test() {
    const auto dit=owners(Stage::dit,true);
    assert(dit[0] && dit[1] && dit[2] && !dit[3]);
    const auto remote=owners(Stage::gemma,true);
    assert(!remote[0] && !remote[1] && !remote[2] && remote[3]);
    const auto local=owners(Stage::gemma,false);
    assert(local[0] && !local[3]);
    for(auto s : {Stage::connectors,Stage::decoder}) {
        const auto o=owners(s,true);
        assert(o[0] && !o[1] && !o[2] && !o[3]);
    }
}

void reduced_linear_test() {
    // Split K into independent inputs, then one sum and one output bias.
    const std::array<std::vector<float>,3> partial{{{6.f,12.f},{7.f,13.f},{8.f,14.f}}};
    const std::vector<float> bias{.5f,-.5f};
    const auto got=reduce_partials(partial,bias);
    assert(got==std::vector<float>({21.5f,38.5f}));
    const float wrong_bias_repeated=6+7+8+3*.5f;
    assert(got[0]!=wrong_bias_repeated);
    auto bad=partial;
    bad[1].pop_back();
    refuse([&]{ reduce_partials(bad,bias); });
    refuse([&]{ reduce_partials(partial,{}); });
}

void mailbox_test() {
    ConditioningMailbox mb;
    const Ticket t{99,10,2,8,2};
    std::vector<std::uint8_t> data(32);
    for(std::size_t i=0;i<data.size();++i) data[i]=std::uint8_t(i+13);
    mb.publish(t,data);
    assert(mb.pending());
    refuse([&]{ mb.publish(t,data); }); // can't overwrite unfinished transfer
    Ticket stale=t; stale.epoch=9;
    refuse([&]{ mb.receive(stale); }); // cannot cross request epochs
    Ticket wrong_shape=t; wrong_shape.columns=4;
    refuse([&]{ mb.receive(wrong_shape); });
    Ticket wrong_id=t; wrong_id.request=100;
    refuse([&]{ mb.receive(wrong_id); });
    assert(mb.receive(t)==data);
    assert(!mb.pending());
    refuse([&]{ mb.receive(t); }); // replay consumed transfer
    mb.publish(t,data);
    mb.cancel();
    refuse([&]{ mb.receive(t); });
    refuse([&]{ mb.publish(Ticket{0,10,2,8,2},data); });
    refuse([&]{ mb.publish(t,std::vector<std::uint8_t>{1}); });
}

int main() {
    schedule_test();
    ownership_test();
    reduced_linear_test();
    mailbox_test();
    std::cout << "LTX TP3 schedule/handoff reference: 4 groups passed\n";
}
