#include "bc_engine_settings.h"

#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace {
int callback_case = 0;
template <class T>
concept RawSymbolAccessible = requires(const T& value) { value.function_; };
using SettingsHook = bc::serving::callback::GuardedSymbol<int(BcEngineSettings*)>;
static_assert(!std::is_invocable_v<SettingsHook, BcEngineSettings*>);
static_assert(!RawSymbolAccessible<SettingsHook>);
template <class T>
concept RawSymbolTypeNameable = requires { typename T::Pointer; };
static_assert(!RawSymbolTypeNameable<SettingsHook>);
} // namespace

extern "C" int bc_serving_engine_settings(BcEngineSettings* settings) {
    if (callback_case == 1)
        throw std::runtime_error("injected symbol exception");
    if (callback_case == 2)
        throw 17; // NOLINT(bugprone-std-exception-baseclass): deliberate nonstandard injection
    settings->manifest = callback_case == 3 ? 0U : 1U;
    if (callback_case == 4)
        settings->struct_size = 0U;
    settings->gemv = 1U;
    return callback_case == 5 ? 0 : 1;
}

int main() {
    auto missing = bc::serving::callback::GuardedSymbol<int(BcEngineSettings*)>::lookup(
        "not_an_exported_callback_for_test");
    BcEngineSettings target{};
    assert(!missing.invoke(&target));

    for (int i = 0; i <= 5; ++i) {
        callback_case = i;
        const auto settings = bc_engine_settings_snapshot();
        if (i == 0) {
            assert(settings.manifest == 1U && settings.gemv == 1U);
        } else {
            assert(settings.manifest == 0U && settings.gemv == 0U);
        }
    }
    return 0;
}
