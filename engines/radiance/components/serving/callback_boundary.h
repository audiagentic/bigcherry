#pragma once
/* Sole invocation boundary for injected serving-session callbacks.
 * The stored callable cannot be retrieved or directly invoked by the owner.
 * Compatible with the power component's v2 typed failure model.
 */
#include "v2/error.h"
#include <exception>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace bc::serving::session {

template <class Signature> class Callback;

template <class Return, class... Args> class Callback<Return(Args...)> {
  public:
    // v2::Status is already a typed success/error; never nest it in Result.
    using Outcome = std::conditional_t<std::is_void_v<Return> || std::is_same_v<Return, v2::Status>,
                                       v2::Status, v2::Result<Return>>;

    Callback() = default;
    Callback(const Callback&) = default;
    Callback(Callback&&) noexcept = default;
    Callback& operator=(const Callback&) = default;
    Callback& operator=(Callback&&) noexcept = default;

    // Exact return type: never discard a returned Status through a void callback.
    // Must remain C++17-compatible for the HIP17 moe-remote include path.
    template <class Fn, std::enable_if_t<!std::is_same_v<std::decay_t<Fn>, Callback> &&
                                             std::is_invocable_v<Fn&, Args...> &&
                                             std::is_same_v<std::invoke_result_t<Fn&, Args...>, Return>,
                                         int> = 0>
    Callback(Fn&& fn) : callable_(std::forward<Fn>(fn)) {}

    // Exact return type: never discard a returned Status through a void callback.
    // Must remain C++17-compatible for the HIP17 moe-remote include path.
    template <class Fn, std::enable_if_t<!std::is_same_v<std::decay_t<Fn>, Callback> &&
                                             std::is_invocable_v<Fn&, Args...> &&
                                             std::is_same_v<std::invoke_result_t<Fn&, Args...>, Return>,
                                         int> = 0>
    Callback& operator=(Fn&& fn) {
        callable_ = std::forward<Fn>(fn);
        return *this;
    }

    explicit operator bool() const noexcept { return static_cast<bool>(callable_); }

    [[nodiscard]] Outcome invoke(Args... args) const {
        if (!callable_)
            return v2::make_error(v2::ConfigErrorCode::invalid_state, "callback",
                                  "required callback was not installed");
        try {
            if constexpr (std::is_void_v<Return>) {
                callable_(std::forward<Args>(args)...);
                return std::nullopt;
            } else {
                return callable_(std::forward<Args>(args)...);
            }
        } catch (const std::exception& error) {
            return v2::make_error(v2::ConfigErrorCode::io_failure, "callback",
                                  std::string("callback threw: ") + error.what());
        } catch (...) {
            return v2::make_error(v2::ConfigErrorCode::io_failure, "callback",
                                  "callback threw a non-standard exception");
        }
    }

  private:
    std::function<Return(Args...)> callable_;
};

} // namespace bc::serving::session
