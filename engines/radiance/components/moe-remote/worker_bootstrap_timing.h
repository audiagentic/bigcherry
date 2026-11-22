// Startup-only stage timing of one helper's expert bootstrap. The helper loads its experts one runtime layer at a time
// after the engine publishes each layer's map (worker_bootstrap.h), and a layer's map is not acknowledged until its
// experts and pointer tables are on the device, so every stage here sits on the engine's startup critical path. Spans
// are measured with the monotonic clock and aggregated per (layer, stage) as bytes, calls, total and max nanoseconds;
// the whole table is emitted once, as one structured log line, when the bootstrap ends. Nothing here is on a decode or
// request path: the timer exists only for the duration of the helper's bootstrap and is dropped afterwards.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace r3::bootstrap_timing {

/* MapWait: the helper waiting for the engine to publish a layer's runtime map (engine-side latency, not helper work).
 * Read: buffered pread of source planes. Relayout: the library's CPU relayout of W4 planes. H2D: the synchronous
 * hipMemcpy of each laid-out plane. Table: building and uploading the layer's pointer tables. LayerLoad: the whole
 * per-layer load (contains Read, Relayout, H2D and Table plus slot admission). Seal: the controller seal wait and
 * seal (recorded once per helper, not per layer). The bootstrap does not stage through pinned host memory (only the
 * live-move path does), so there is no pinned-staging stage. */
enum class Stage : uint8_t { MapWait, Read, Relayout, H2D, Table, LayerLoad, Seal, kCount };
constexpr size_t kStages = size_t(Stage::kCount);
constexpr size_t kLayerStages = size_t(Stage::Seal);   /* the stages kept per layer; Seal is one per helper */

inline const char* stage_name(Stage s) {
    switch (s) {
        case Stage::MapWait: return "map_wait";
        case Stage::Read: return "read";
        case Stage::Relayout: return "relayout";
        case Stage::H2D: return "h2d";
        case Stage::Table: return "table";
        case Stage::LayerLoad: return "layer_load";
        case Stage::Seal: return "seal";
        default: return "?";
    }
}

struct Span {
    uint64_t bytes = 0, calls = 0, total_ns = 0, max_ns = 0;
    void add(uint64_t b, uint64_t ns) {
        bytes += b; ++calls; total_ns += ns;
        max_ns = std::max(max_ns, ns);
    }
};

inline uint64_t now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

class Timer {
public:
    explicit Timer(uint32_t layers) : layers_(layers), spans_(size_t(layers) * kStages) {}

    /* The layer the helper is loading: spans recorded below the per-layer code (plane reads, copies) land on it. */
    void set_layer(uint32_t layer) { layer_ = layer; }
    uint32_t layer() const { return layer_; }

    void record(Stage s, uint64_t bytes, uint64_t ns) { record_at(layer_, s, bytes, ns); }
    void record_at(uint32_t layer, Stage s, uint64_t bytes, uint64_t ns) {
        if (layer >= layers_ || s == Stage::Seal) return;
        spans_[size_t(layer) * kStages + size_t(s)].add(bytes, ns);
    }
    const Span& at(uint32_t layer, Stage s) const { return spans_[size_t(layer) * kStages + size_t(s)]; }
    void record_seal(uint64_t ns) { seal_.add(0, ns); }

    Span total(Stage s) const {
        if (s == Stage::Seal) return seal_;
        Span t;
        for (uint32_t l = 0; l < layers_; ++l) {
            const Span& x = at(l, s);
            t.bytes += x.bytes; t.calls += x.calls; t.total_ns += x.total_ns;
            t.max_ns = std::max(t.max_ns, x.max_ns);
        }
        return t;
    }
    uint32_t layers() const { return layers_; }

    /* One line, no embedded newline: `r3_moe_worker: bootstrap_stage_timing {json}`. "totals" aggregates each stage
     * over layers; "layers" lists only layers with a recorded span, as {"layer":L,"<stage>":[bytes,calls,total_ns,max_ns]}. */
    std::string line(uint32_t helper) const {
        std::string o = "r3_moe_worker: bootstrap_stage_timing {\"helper\":" + std::to_string(helper) +
                        ",\"layers\":" + std::to_string(layers_) + ",\"totals\":{";
        for (size_t s = 0; s < kStages; ++s) {
            if (s) o += ',';
            o += '"'; o += stage_name(Stage(s)); o += "\":"; o += tuple(total(Stage(s)));
        }
        o += "},\"per_layer\":[";
        bool first = true;
        for (uint32_t l = 0; l < layers_; ++l) {
            bool any = false;
            for (size_t s = 0; s < kLayerStages; ++s) any |= at(l, Stage(s)).calls != 0;
            if (!any) continue;
            if (!first) o += ',';
            first = false;
            o += "{\"layer\":" + std::to_string(l);
            for (size_t s = 0; s < kLayerStages; ++s) {
                const Span& x = at(l, Stage(s));
                if (!x.calls) continue;
                o += ",\""; o += stage_name(Stage(s)); o += "\":"; o += tuple(x);
            }
            o += '}';
        }
        o += "]}";
        return o;
    }

private:
    static std::string tuple(const Span& x) {
        return "[" + std::to_string(x.bytes) + "," + std::to_string(x.calls) + "," + std::to_string(x.total_ns) +
               "," + std::to_string(x.max_ns) + "]";
    }
    uint32_t layers_;
    uint32_t layer_ = 0;
    std::vector<Span> spans_;
    Span seal_;
};

/* Times one span on the timer's current layer; a null timer (live moves) costs only the null check. */
class Scope {
public:
    Scope(Timer* t, Stage s, uint64_t bytes = 0) : t_(t), s_(s), bytes_(bytes), start_(t ? now_ns() : 0) {}
    ~Scope() { if (t_) t_->record(s_, bytes_, now_ns() - start_); }
    void add_bytes(uint64_t b) { bytes_ += b; }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
private:
    Timer* t_;
    Stage s_;
    uint64_t bytes_, start_;
};

}  // namespace r3::bootstrap_timing
