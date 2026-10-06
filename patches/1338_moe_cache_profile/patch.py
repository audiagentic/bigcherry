"""1338: profile-pinned MoE expert cache (frequency residency borrowed from Strata, on top of 1337 / upstream #29887).

1337's cache is a pure LRU over (layer, expert) slots and only serves ubatches of up to 32 tokens: a large batch
uses most experts of a layer and would evict what generation needs, so prefill still uploads every selected expert
of every host layer per micro-batch (1336).

Routing is strongly skewed: with half of all (layer, expert) pairs resident, a frequency ranking covers 91-97% of
the routed pairs of an unseen request, where whole layers (--n-cpu-moe) cover 50% (MET01, routing-skew.py). This
patch uses that:

  1. BIGCHERRY_MOE_CACHE_PROFILE=<file> reads a Strata profile (`STRP`: (layer, expert) pairs ranked by routing
     frequency; tools/make_profile.py of Strata writes it, and so does this patch) and at start-up uploads the top
     pairs into BIGCHERRY_MOE_CACHE_PIN_PCT (default 85) percent of each group's slots. Pinned slots are never
     evicted; the rest of the slots stay an LRU.
  2. With a pinned set, large batches use the cache too (BIGCHERRY_MOE_CACHE_LARGE=0 turns that off): hits are
     computed from the resident experts and only the misses are uploaded, into the LRU tail, so a prefill sweep
     cannot evict the hot set. A batch whose unpinned experts do not fit the tail falls back to 1336's upload.
  3. BIGCHERRY_MOE_CACHE_PROFILE_OUT=<file> counts the routed experts the cache sees and writes a profile at exit
     (ranked by count, then the pairs never routed, interleaved across layers), in the same format.

Nothing changes without the two variables: no pins, no large batches, no counting.
"""
import re as _re

from bigcherry.patcher import Edit, EnvDoc, FilePatch

GROUP = "core"
STATE = "untested"

_A_INC = "#include <cstring>\n#include <stdexcept>\n"
_N_INC = "#include <cstdio>   // BigCherry 1338\n#include <cstdlib>\n#include <cstring>\n#include <stdexcept>\n#include <string>\n"

_A_LRU_FIELDS = "    std::vector<uint32_t> seen; // [n_expert]\n"
_N_LRU_FIELDS = r"""    // BigCherry 1338: pinned slots are outside the LRU list and are never evicted
    std::vector<uint8_t> pinned; // [n_slots]
    int32_t n_pinned = 0;

    // gives key the least recently used slot for good; -1 when the key is cached or one unpinned slot is left
    int32_t pin(int32_t key) {
        if (head < 0 || head == tail || slot_of[key] >= 0) {
            return -1;
        }
        const int32_t s = head;
        head = next[s];
        prev[head] = -1;
        prev[s] = -1;
        next[s] = -1;
        if (key_of[s] >= 0) {
            slot_of[key_of[s]] = -1;
        }
        key_of[s] = key;
        slot_of[key] = s;
        pinned[s] = 1;
        n_pinned++;
        return s;
    }

""" + _A_LRU_FIELDS

_A_LRU_INIT = "        seen.assign(n_expert, 0);\n    }\n"
_N_LRU_INIT = "        seen.assign(n_expert, 0);\n        pinned.assign(n_slots, 0); // BigCherry 1338\n        n_pinned = 0;\n    }\n"

_A_LRU_TOUCH = "    void touch(int32_t s) {\n        if (s == tail) {\n            return;\n        }\n"
_N_LRU_TOUCH = "    void touch(int32_t s) {\n        if (s == tail || pinned[s]) { // BigCherry 1338: a pinned slot is not in the list\n            return;\n        }\n"

_A_LRU_PLAN = "        if (uniq.size() > (size_t) n_slots) {\n            return false;\n        }\n"
_N_LRU_PLAN = _A_LRU_PLAN + r"""        if (n_pinned > 0) {
            // BigCherry 1338: everything that is not a pinned hit has to fit the unpinned slots together, or an
            // eviction would take an expert this same batch uses
            size_t bc_unpinned = 0;
            for (int32_t e : uniq) {
                const int32_t s = slot_of[(size_t) il*n_expert + e];
                if (s < 0 || !pinned[s]) {
                    bc_unpinned++;
                }
            }
            if (bc_unpinned > (size_t) (n_slots - n_pinned)) {
                return false;
            }
        }
"""

_A_MEMBERS = "    static constexpr int64_t max_batch = 32;\n"
_N_MEMBERS = _A_MEMBERS + r"""
    // BigCherry 1338: profile-pinned residency
    bool bc_large_batches = false;   // large batches may use the cache (only with a pinned set)
    int32_t bc_n_expert = 0;
    size_t bc_pinned = 0;
    size_t bc_pinned_bytes = 0;
    std::vector<uint32_t> bc_freq;   // [n_layer*n_expert] routed count, when a profile is to be written
    std::string bc_profile_out;
"""

_A_RESOLVE = "        if (n_tokens > max_batch || std::min("
_N_RESOLVE = "        if ((n_tokens > max_batch && !bc_large_batches) || std::min("

_A_PREPARE = "        l.planned_ids.assign(ids, ids + n_ids);\n"
_N_PREPARE = _A_PREPARE + r"""        if (!bc_freq.empty()) { // BigCherry 1338: routing counts for the profile
            for (size_t i = 0; i < n_ids; ++i) {
                bc_freq[(size_t) entry->il*bc_n_expert + ids[i]]++;
            }
        }
"""

_A_CTOR_LOG = '        LLAMA_LOG_INFO("%s: %10s MoE cache size = '
_N_CTOR_LOG = "        bc_prewarm(n_expert); // BigCherry 1338\n\n" + _A_CTOR_LOG

_A_DTOR = "    ~impl() {\n        log_stats();\n    }\n"
_N_DTOR = r"""    // BigCherry 1338: Strata profile, `STRP`, u32 version, n_layer, n_expert, slots, n, then n (u16 layer, u16 expert)
    // pairs ranked by routing frequency, then an n_layer x n_expert i32 table of each pair's rank
    static std::vector<std::pair<uint16_t, uint16_t>> bc_read_profile(const char * path, uint32_t n_layer, uint32_t n_expert) {
        FILE * f = fopen(path, "rb");
        if (f == nullptr) {
            throw std::runtime_error(std::string("BIGCHERRY_MOE_CACHE_PROFILE: cannot open ") + path);
        }
        char magic[4];
        uint32_t head[5];
        std::vector<std::pair<uint16_t, uint16_t>> ranked;
        bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, "STRP", 4) == 0 && fread(head, sizeof(uint32_t), 5, f) == 5;
        if (ok && (head[1] != n_layer || head[2] != n_expert)) {
            fclose(f);
            throw std::runtime_error(std::string("BIGCHERRY_MOE_CACHE_PROFILE: layer / expert counts do not match the model: ") + path);
        }
        if (ok) {
            ranked.resize(head[4]);
            for (auto & p : ranked) {
                uint16_t le[2];
                ok = ok && fread(le, sizeof(uint16_t), 2, f) == 2 && le[0] < n_layer && le[1] < n_expert;
                p = { le[0], le[1] };
            }
        }
        fclose(f);
        if (!ok) {
            throw std::runtime_error(std::string("BIGCHERRY_MOE_CACHE_PROFILE: not a valid profile: ") + path);
        }
        return ranked;
    }

    void bc_write_profile() const {
        const size_t n_layer = layers.size();
        const size_t n_pairs = n_layer*bc_n_expert;
        std::vector<uint32_t> order;
        for (size_t k = 0; k < n_pairs; ++k) {
            if (bc_freq[k] > 0) {
                order.push_back(k);
            }
        }
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return bc_freq[a] > bc_freq[b]; });
        const size_t n_routed = order.size();
        for (int32_t e = 0; e < bc_n_expert; ++e) { // the pairs never routed, interleaved across the layers
            for (size_t il = 0; il < n_layer; ++il) {
                if (bc_freq[il*bc_n_expert + e] == 0) {
                    order.push_back(il*bc_n_expert + e);
                }
            }
        }
        std::vector<int32_t> rank(n_pairs, -1);
        for (size_t r = 0; r < order.size(); ++r) {
            rank[order[r]] = (int32_t) r;
        }
        FILE * f = fopen(bc_profile_out.c_str(), "wb");
        if (f == nullptr) {
            LLAMA_LOG_WARN("llama_moe_cache: cannot write the profile %s\n", bc_profile_out.c_str());
            return;
        }
        const uint32_t head[5] = { 1, (uint32_t) n_layer, (uint32_t) bc_n_expert, (uint32_t) order.size(), (uint32_t) order.size() };
        fwrite("STRP", 1, 4, f);
        fwrite(head, sizeof(uint32_t), 5, f);
        for (uint32_t k : order) {
            const uint16_t le[2] = { (uint16_t) (k / bc_n_expert), (uint16_t) (k % bc_n_expert) };
            fwrite(le, sizeof(uint16_t), 2, f);
        }
        fwrite(rank.data(), sizeof(int32_t), rank.size(), f);
        fclose(f);
        LLAMA_LOG_INFO("llama_moe_cache: wrote the profile %s, %zu routed pairs of %zu\n", bc_profile_out.c_str(), n_routed, n_pairs);
    }

    // uploads the top pairs of the profile into pinned slots and switches large batches on
    void bc_prewarm(int32_t n_expert) {
        bc_n_expert = n_expert;
        const char * out = getenv("BIGCHERRY_MOE_CACHE_PROFILE_OUT");
        if (out != nullptr && *out != '\0') {
            bc_profile_out = out;
            bc_freq.assign(layers.size()*n_expert, 0);
        }
        const char * path = getenv("BIGCHERRY_MOE_CACHE_PROFILE");
        if (path == nullptr || *path == '\0' || no_alloc) {
            return;
        }
        const auto ranked = bc_read_profile(path, (uint32_t) layers.size(), (uint32_t) n_expert);
        const char * pct_env = getenv("BIGCHERRY_MOE_CACHE_PIN_PCT");
        const int pct = pct_env != nullptr ? std::max(0, std::min(100, atoi(pct_env))) : 85;

        std::vector<int32_t> group_of(layers.size(), -1);
        std::vector<int64_t> quota(groups.size(), 0);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (groups[ig].n_slots == 0) {
                continue;
            }
            // the tail keeps room for the experts of one generation ubatch at least
            quota[ig] = std::max<int64_t>(0, std::min<int64_t>((int64_t) groups[ig].n_slots*pct/100, groups[ig].n_slots - 8*n_expert_used));
            for (int32_t il : groups[ig].layers) {
                group_of[il] = ig;
            }
        }
        for (const auto & p : ranked) {
            const int32_t ig = group_of[p.first];
            if (ig < 0 || quota[ig] <= 0) {
                continue;
            }
            const int32_t s = groups[ig].lru.pin((int32_t) p.first*n_expert + p.second);
            if (s < 0) {
                continue;
            }
            quota[ig]--;
            for (int32_t ib : layers[p.first].bindings) {
                const binding & b = bindings[ib];
                const size_t expert_size = b.src->nb[2];
                ggml_backend_tensor_set(b.bank, (const uint8_t *) b.src->data + p.second*expert_size, s*expert_size, expert_size);
                bc_pinned_bytes += expert_size;
            }
            bc_pinned++;
        }
        const char * large = getenv("BIGCHERRY_MOE_CACHE_LARGE");
        bc_large_batches = bc_pinned > 0 && (large == nullptr || atoi(large) != 0);
        LLAMA_LOG_INFO("llama_moe_cache: profile %s: %zu experts pinned (%.2f MiB, %d%% of the slots), large batches %s\n",
            path, bc_pinned, bc_pinned_bytes/1024.0/1024.0, pct, bc_large_batches ? "use the cache" : "bypass the cache");
    }

    ~impl() {
        log_stats();
        if (!bc_profile_out.empty()) {
            bc_write_profile();
        }
        if ((bc_pinned > 0 || !bc_profile_out.empty()) && getenv("BIGCHERRY_PATCH_TRACE") != nullptr) {
            fprintf(stderr, "BIGCHERRY_PATCH_HIT patch=1338_moe_cache_profile pinned=%zu pinned_bytes=%zu small_hits=%zu small_misses=%zu "
                    "small_bytes=%zu large_hits=%zu large_misses=%zu large_bytes=%zu\n", bc_pinned, bc_pinned_bytes,
                    stats_small.hits, stats_small.misses, stats_small.bytes, stats_large.hits, stats_large.misses, stats_large.bytes);
        }
    }
"""

PATCHES = [
    FilePatch(
        path="src/llama-moe-cache.cpp",
        description="1338: frequency profile for the MoE expert cache - pinned hot set, large batches through the cache, profile writer",
        language="none",
        edits=(
            Edit(id="cache-profile-includes", anchor=_re.escape(_A_INC), mode="replace", text=_N_INC,
                 guard=r"#include <cstdio>   // BigCherry 1338", rationale="Standard includes of the cache source.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="cache-profile-lru-pin", anchor=_re.escape(_A_LRU_FIELDS), mode="replace", text=_N_LRU_FIELDS,
                 guard=r"BigCherry 1338: pinned slots are outside the LRU list", rationale="Fields of the LRU map.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="cache-profile-lru-init", anchor=_re.escape(_A_LRU_INIT), mode="replace", text=_N_LRU_INIT,
                 guard=r"pinned\.assign\(n_slots, 0\); // BigCherry 1338", rationale="End of the LRU map's init.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="cache-profile-lru-touch", anchor=_re.escape(_A_LRU_TOUCH), mode="replace", text=_N_LRU_TOUCH,
                 guard=r"s == tail \|\| pinned\[s\]", rationale="Head of touch(): a pinned slot has no list links.",
                 expect_matches=1, max_span_lines=5),
            Edit(id="cache-profile-lru-plan", anchor=_re.escape(_A_LRU_PLAN), mode="replace", text=_N_LRU_PLAN,
                 guard=r"bc_unpinned > \(size_t\) \(n_slots - n_pinned\)",
                 rationale="After upstream's capacity check in plan(), before any slot is touched.", expect_matches=1,
                 max_span_lines=4),
            Edit(id="cache-profile-members", anchor=_re.escape(_A_MEMBERS), mode="replace", text=_N_MEMBERS,
                 guard=r"BigCherry 1338: profile-pinned residency", rationale="Members of the cache implementation.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="cache-profile-resolve", anchor=_re.escape(_A_RESOLVE), mode="replace", text=_N_RESOLVE,
                 guard=r"n_tokens > max_batch && !bc_large_batches", rationale="The batch-size gate of resolve().",
                 expect_matches=1, max_span_lines=2),
            Edit(id="cache-profile-count", anchor=_re.escape(_A_PREPARE), mode="replace", text=_N_PREPARE,
                 guard=r"BigCherry 1338: routing counts for the profile",
                 rationale="prepare(), where a layer's ids are planned once per graph.", expect_matches=1, max_span_lines=2),
            Edit(id="cache-profile-prewarm-call", anchor=_re.escape(_A_CTOR_LOG), mode="replace", text=_N_CTOR_LOG,
                 guard=r"bc_prewarm\(n_expert\); // BigCherry 1338",
                 rationale="End of the constructor, after the banks are allocated and bound.", expect_matches=1,
                 max_span_lines=2),
            Edit(id="cache-profile-methods", anchor=_re.escape(_A_DTOR), mode="replace", text=_N_DTOR,
                 guard=r"static std::vector<std::pair<uint16_t, uint16_t>> bc_read_profile\(",
                 rationale="The implementation's destructor: profile reader / writer / prewarm before it, marker in it.",
                 expect_matches=1, max_span_lines=4),
        ),
    ),
]

ENV_DOCS = (
    EnvDoc("BIGCHERRY_MOE_CACHE_PROFILE", "<file>", "(unset)",
           "with --moe-cache-mib: Strata-format frequency profile; its top (layer, expert) pairs are uploaded at "
           "start-up into pinned cache slots and large batches then use the cache"),
    EnvDoc("BIGCHERRY_MOE_CACHE_PIN_PCT", "0..100", "85",
           "share of the cache slots given to the pinned profile set; the rest stays an LRU"),
    EnvDoc("BIGCHERRY_MOE_CACHE_LARGE", "0|1", "1",
           "with a pinned set: 0 keeps ubatches above 32 tokens off the cache (1337's behaviour)"),
    EnvDoc("BIGCHERRY_MOE_CACHE_PROFILE_OUT", "<file>", "(unset)",
           "with --moe-cache-mib: count the routed experts the cache sees and write a profile at exit"),
)
