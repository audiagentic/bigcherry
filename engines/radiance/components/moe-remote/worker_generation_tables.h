#pragma once
// Two immutable per-generation pointer-table banks. The GPU uses table_index;
// the card-free model applies the same indexing to fake pointer identifiers.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace r3::worker_tables {
constexpr uint32_t kGenerations = 2; // v23 Request::generation_slot protocol
constexpr bool valid_generation(uint32_t slot) { return slot < kGenerations; }

constexpr size_t table_index(uint32_t generation, uint32_t layer, uint32_t layers) {
    return size_t(generation) * layers + layer;
}

template<class Pointer>
class Model {
    uint32_t layers_, experts_;
    std::vector<Pointer> bindings_;
    bool ready_[kGenerations]{};
public:
    Model(uint32_t layers, uint32_t experts)
        : layers_(layers), experts_(experts),
          bindings_(size_t(kGenerations) * layers * experts) {}
    // A bank is built in full before it can serve. A published bank is immutable
    // until retirement of all its generation's requests is proven externally.
    bool build(uint32_t generation, const std::vector<Pointer>& bindings) {
        if (!valid_generation(generation) || ready_[generation] ||
            bindings.size() != size_t(layers_) * experts_) return false;
        const size_t off = table_index(generation, 0, layers_) * experts_;
        for (size_t i = 0; i < bindings.size(); ++i) bindings_[off + i] = bindings[i];
        ready_[generation] = true;
        return true;
    }
    bool select(uint32_t generation, uint32_t layer, uint32_t expert, Pointer* out) const {
        if (!out || !valid_generation(generation) || !ready_[generation] ||
            layer >= layers_ || expert >= experts_) return false;
        *out = bindings_[table_index(generation, layer, layers_) * experts_ + expert];
        return true;
    }
    bool retire(uint32_t generation, bool all_references_retired) {
        if (!valid_generation(generation) || !ready_[generation] || !all_references_retired)
            return false;
        ready_[generation] = false;
        return true;
    }
};
} // namespace r3::worker_tables
