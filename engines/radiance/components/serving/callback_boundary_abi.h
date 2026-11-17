#pragma once

// C-ABI dynamic symbol hooks use the same guarded-callable family as C++
// injected hooks, without importing the serving v2 or C++20 runtime into
// HIP-facing headers. No raw function pointer can be named by a caller.
#include <dlfcn.h>

#include <optional>
#include <utility>

namespace bc::serving::callback {

template <class Signature> class GuardedSymbol;

template <class R, class... Args> class GuardedSymbol<R(Args...)> final {
  public:
    static GuardedSymbol lookup(const char* name) noexcept {
        return GuardedSymbol(reinterpret_cast<Pointer>(::dlsym(RTLD_DEFAULT, name)));
    }

    // Missing symbols and thrown C-ABI implementations are both typed absence.
    // C-ABI settings readers already define absence as inert, rather than fatal.
    std::optional<R> invoke(Args... args) const noexcept {
        if (!function_)
            return std::nullopt;
        try {
            return function_(std::forward<Args>(args)...);
        } catch (...) {
            return std::nullopt;
        }
    }

  private:
    using Pointer = R (*)(Args...);
    explicit GuardedSymbol(Pointer function) noexcept : function_(function) {}
    Pointer function_ = nullptr;
};

} // namespace bc::serving::callback
