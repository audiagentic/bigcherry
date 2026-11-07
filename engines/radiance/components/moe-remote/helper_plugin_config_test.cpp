#ifdef NDEBUG
#undef NDEBUG
#endif
#include "worker_plugin_config.h"
#include <cassert>
#ifdef NDEBUG
#error "assert-based test built with NDEBUG: its assertions would not run"
#endif
#include <sys/wait.h>
#include <unistd.h>

template<class T> T symbol(void* library, const char* name) {
    const auto p = reinterpret_cast<T>(dlsym(library, name));
    assert(p);
    return p;
}
int main(int argc, char** argv) {
    assert(argc == 2);
    // Each helper is a new process with exactly one immutable plugin configuration.
    for (unsigned variant = 0; variant != 49; ++variant) {
        const pid_t pid = fork();
        assert(pid >= 0);
        if (pid == 0) {
            void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
            assert(library);
            auto configure = symbol<int (*)(const BcR4dHelperConfig*)>(library, "bc_r4d_configure_helper");
            auto current = symbol<const BcR4dHelperConfig* (*)()>(library, "bc_r4d_helper_config");
            auto read = symbol<BcR4dHelperConfig (*)()>(library, "test_moe_settings");
            auto trace = symbol<int (*)()>(library, "test_refusal_trace");
            auto env_reads = symbol<unsigned (*)()>(library, "test_env_reads");
            assert(!current() && env_reads() == 0);
            if (variant == 48) {
                // Engine path still has its original environment semantics.
                const auto engine = read();
                assert(!engine.native && !engine.fp16 && !engine.ahalf && engine.moe_probe == 0);   // R3_MOE_PROBE (a timing-only ablation) is no longer read from the environment (RSC12)
                assert(trace() == 1 && env_reads() == 4);
            } else {
                BcR4dHelperConfig bad{};
                assert(configure(nullptr) == -1 && configure(&bad) == -1);
                bad.struct_size = sizeof(bad) + 4;
                assert(configure(&bad) == -1);
                bad.struct_size = sizeof(bad);
                bad.native = 2;
                assert(configure(&bad) == -1);
                bad.native = 0;
                bad.moe_probe = 3;
                assert(configure(&bad) == -1 && !current());
                bc::serving::HelperSettings settings;
                settings.native = variant & 1;
                settings.fp16 = variant & 2;
                settings.ahalf = variant & 4;
                settings.moe_probe = (variant / 8) % 3;
                settings.refusal_trace = variant / 24;
                assert(r3_serving::configure_plugin(library, settings));
                const auto installed = *current();
                // The library copied the input; later mutation cannot change it.
                settings.native = !settings.native;
                const auto selected = read();
                assert(selected.native == (variant & 1 ? 1u : 0u));
                assert(selected.fp16 == (variant & 2 ? 1u : 0u));
                assert(selected.ahalf == (variant & 4 ? 1u : 0u));
                assert(selected.moe_probe == (variant / 8) % 3);
                assert(trace() == int(variant / 24));
                assert(env_reads() == 0);
                assert(configure(&installed) == -1);
                assert(!r3_serving::configure_plugin(library, settings));
                assert(current()->native == installed.native);
            }
            dlclose(library);
            _exit(0);
        }
        int status = 0;
        assert(waitpid(pid, &status, 0) == pid);
        assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    // A plugin without the typed interface is refused rather than used unconfigured.
    void* no_interface = dlopen(nullptr, RTLD_NOW | RTLD_LOCAL);
    assert(no_interface);
    assert(!r3_serving::configure_plugin(no_interface, bc::serving::HelperSettings{}));
    dlclose(no_interface);
}
