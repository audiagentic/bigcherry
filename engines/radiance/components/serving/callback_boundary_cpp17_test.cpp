#include "callback_boundary.h"
#include <type_traits>

// Included by the C++17 HIP helper; never silently discard a hook return.
namespace {
using bc::serving::callback::Guarded;
struct VoidHook {
    void operator()(int) const;
};
struct BoolHook {
    bool operator()(int) const;
};
struct StatusHook {
    bc::serving::v2::Status operator()(int) const;
};
struct WrongArgs {
    void operator()() const;
};
static_assert(std::is_constructible_v<Guarded<void(int)>, VoidHook>);
static_assert(std::is_constructible_v<Guarded<bool(int)>, BoolHook>);
static_assert(!std::is_constructible_v<Guarded<void(int)>, StatusHook>);
static_assert(!std::is_constructible_v<Guarded<void(int)>, BoolHook>);
static_assert(!std::is_constructible_v<Guarded<void(int)>, WrongArgs>);
static_assert(!std::is_constructible_v<Guarded<bool(int)>, VoidHook>);
static_assert(!std::is_invocable_v<Guarded<void(int)>&, int>);
static_assert(!std::is_invocable_v<Guarded<bool(int)>&, int>);
static_assert(std::is_copy_constructible_v<Guarded<void(int)>>);
static_assert(std::is_move_constructible_v<Guarded<void(int)>>);
} // namespace
int main() {
    return 0;
}
