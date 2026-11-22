// The startup stage timer of a helper's expert bootstrap: aggregation per (layer, stage), the one-line log format, the
// scope's monotonic measurement, and that a null timer records nothing. Pure host code.
#include "worker_bootstrap_timing.h"
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace bt = r3::bootstrap_timing;
using bt::Stage;

static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "r3_bootstrap_timing_test: FAILED %s\n", what); std::exit(1); }
}

int main() {
    // Aggregation: bytes, calls and total sum, max is the largest single span, per layer and per stage.
    bt::Timer t(3);
    t.record_at(0, Stage::Read, 100, 10);
    t.record_at(0, Stage::Read, 50, 30);
    t.record_at(0, Stage::Read, 25, 20);
    t.record_at(2, Stage::Read, 7, 5);
    t.record_at(2, Stage::H2D, 9, 4);
    check(t.at(0, Stage::Read).bytes == 175 && t.at(0, Stage::Read).calls == 3 &&
          t.at(0, Stage::Read).total_ns == 60 && t.at(0, Stage::Read).max_ns == 30, "per-layer aggregate");
    check(t.at(1, Stage::Read).calls == 0, "an untouched layer stays empty");
    const bt::Span read = t.total(Stage::Read);
    check(read.bytes == 182 && read.calls == 4 && read.total_ns == 65 && read.max_ns == 30, "total over layers");
    check(t.total(Stage::H2D).calls == 1, "other stages are separate");

    // The current layer routes spans recorded below the per-layer code; out-of-range layers are dropped, not wrapped.
    t.set_layer(1);
    t.record(Stage::Relayout, 8, 12);
    check(t.at(1, Stage::Relayout).total_ns == 12, "record uses the current layer");
    t.record_at(3, Stage::Read, 1, 1);
    check(t.total(Stage::Read).calls == 4, "out-of-range layer ignored");

    // Seal is one span per helper, not per layer.
    t.record_at(0, Stage::Seal, 1, 1);
    check(t.total(Stage::Seal).calls == 0, "seal is not recorded per layer");
    t.record_seal(77);
    check(t.total(Stage::Seal).total_ns == 77 && t.total(Stage::Seal).calls == 1, "seal span");

    // The log line: one line, fixed key order, totals for every stage, only layers with spans.
    bt::Timer f(2);
    f.record_at(0, Stage::MapWait, 0, 5);
    f.record_at(0, Stage::Read, 100, 10);
    f.record_at(0, Stage::Read, 50, 30);
    f.record_at(1, Stage::H2D, 9, 4);
    f.record_seal(3);
    const std::string expect =
        "r3_moe_worker: bootstrap_stage_timing {\"helper\":2,\"layers\":2,\"totals\":{"
        "\"map_wait\":[0,1,5,5],\"read\":[150,2,40,30],\"relayout\":[0,0,0,0],\"h2d\":[9,1,4,4],\"table\":[0,0,0,0],"
        "\"layer_load\":[0,0,0,0],\"seal\":[0,1,3,3]},\"per_layer\":["
        "{\"layer\":0,\"map_wait\":[0,1,5,5],\"read\":[150,2,40,30]},{\"layer\":1,\"h2d\":[9,1,4,4]}]}";
    check(f.line(2) == expect, "log line format");
    check(f.line(2).find('\n') == std::string::npos, "one line");
    check(bt::Timer(4).line(0).find("\"per_layer\":[]}") != std::string::npos, "empty timer lists no layers");

    // Scope: measures monotonic elapsed time on the current layer; a null timer is inert.
    bt::Timer s(2);
    s.set_layer(1);
    {
        bt::Scope span(&s, Stage::LayerLoad, 5);
        span.add_bytes(6);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const bt::Span& ll = s.at(1, Stage::LayerLoad);
    check(ll.calls == 1 && ll.bytes == 11 && ll.total_ns >= 20000000ull && ll.total_ns < 4000000000ull &&
          ll.max_ns == ll.total_ns, "scope records elapsed ns and bytes");
    { bt::Scope none(nullptr, Stage::Read, 1); }
    check(s.total(Stage::Read).calls == 0, "null timer records nothing");
    std::puts("r3_bootstrap_timing_test: OK");
}
