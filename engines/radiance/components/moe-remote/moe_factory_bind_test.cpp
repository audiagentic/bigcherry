// The engine's explicit factory bind against the REAL built libr3moe (RUM06 S1): load the plugin library, resolve
// bc_moe_remote_plugin_open on its own handle (what Registry::plugin(...)->handle is), and open generations through the
// engine's actual generation code (core/place/bc_moe_remote.cpp of the patched tree). Host-only: no helper process runs,
// so the plugin refuses the open (no shared segment) -- which must leave nothing bound and nothing mapped. A successful
// open and the plugin's own BUSY (second engine while one holds the helper hook) need a live helper and a card; the
// engine-side BUSY (a bound slot refused before the factory runs) is proven in moe_remote_generation_test.
#include "bc_moe_remote.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

extern "C" int bc_serving_remote_settings(int64_t* rows, uint32_t* skip, uint32_t* shadow, uint32_t* mask) {
    *rows = 4; *skip = *shadow = *mask = 0;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    void* plugin = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
    if (!plugin) { std::fprintf(stderr, "dlopen: %s\n", dlerror()); return 3; }
    // The library is a plugin the loader accepts, and exports the factory the engine looks for.
    assert(dlsym(plugin, "rad_plugin_info") && dlsym(plugin, "rad_plugin_abi_version"));
    assert(dlsym(plugin, BC_MOE_REMOTE_OPEN_SYMBOL));

    // Loading started nothing: nothing is bound, and a refused open leaves nothing bound.
    for (int s = 0; s < BC_MOE_REMOTE_MAX_SLOTS; ++s) assert(!bc_moe_remote_lookup(s));
    for (int i = 0; i < 3; ++i) {
        BcMoeRemoteGeneration* g = nullptr;
        const int rc = bc_moe_remote_open(plugin, 4, 2, 5, &g);
        assert(rc == BC_MOE_REMOTE_REFUSED && !g);
        assert(!bc_moe_remote_lookup(4) && !bc_moe_remote_lookup(5));
    }
    // The factory validates its arguments itself.
    auto factory = reinterpret_cast<BcMoeRemoteOpenFn>(dlsym(plugin, BC_MOE_REMOTE_OPEN_SYMBOL));
    BcMoeRemotePlugin out{};
    BcMoeRemoteOpenArgs bad{sizeof(BcMoeRemoteOpenArgs), 0, 4, 2};
    assert(factory(&bad, &out) == BC_MOE_REMOTE_REFUSED);
    bad = BcMoeRemoteOpenArgs{1, 5, 4, 2};
    assert(factory(&bad, &out) == BC_MOE_REMOTE_REFUSED);
    assert(factory(nullptr, &out) == BC_MOE_REMOTE_REFUSED);

    // A plugin library without the factory is ABSENT while the manifest asks for a hand-off.
    void* other = dlopen("libm.so.6", RTLD_LAZY | RTLD_LOCAL);
    assert(other);
    BcMoeRemoteGeneration* g = nullptr;
    assert(bc_moe_remote_open(other, 4, 2, 5, &g) == BC_MOE_REMOTE_ABSENT && !g);
    dlclose(other);
    // Unloading a plugin that never opened a generation is clean.
    assert(dlclose(plugin) == 0);
    return 0;
}
