#pragma once
// The only invocation mechanism for externally injected C++ callables.
// The std::function is private and has no callable/raw accessor. Both
// std::exception and non-standard exceptions become typed IO failures.
#include "callback_boundary_abi.h"
#include "v2/error.h"

#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace bc::serving::callback {

inline v2::ConfigError thrown(std::string_view path, std::string_view message) {
    return v2::make_error(v2::ConfigErrorCode::io_failure, std::string(path),
                          "injected callable threw: " + std::string(message));
}

class Failure final : public std::runtime_error {
  public:
    explicit Failure(v2::ConfigError error, std::exception_ptr cause = {})
        : std::runtime_error(error.path + ": " + error.message), error_(std::move(error)),
          cause_(std::move(cause)) {}
    const v2::ConfigError& error() const noexcept {
        return error_;
    }
    const std::exception_ptr& cause() const noexcept {
        return cause_;
    }

  private:
    v2::ConfigError error_;
    std::exception_ptr cause_;
};

// Injected virtual interfaces cannot retain raw callsites either. Their
// implementations are invoked through this same conversion boundary.
template <class F> auto invoke_backend(std::string_view path, F&& function) -> decltype(function()) {
    try {
        return std::forward<F>(function)();
    } catch (const std::exception& error) {
        return thrown(path, error.what());
    } catch (...) {
        return thrown(path, "non-standard exception");
    }
}

template <class Signature> class Guarded;

template <class R, class... Args> class Guarded<R(Args...)> final {
  public:
    Guarded() = default;
    template <class F>
        requires(std::is_invocable_r_v<R, F&, Args...> && !std::is_same_v<std::remove_cvref_t<F>, Guarded>)
    Guarded(F&& function) : function_(std::forward<F>(function)) {}

    explicit operator bool() const noexcept {
        return bool(function_);
    }

    v2::Result<R> invoke(std::string_view path, Args... args) const {
        if (!function_)
            return v2::make_error(v2::ConfigErrorCode::invalid_state, std::string(path), "missing callback");
        try {
            return function_(std::forward<Args>(args)...);
        } catch (const std::exception& error) {
            return thrown(path, error.what());
        } catch (...) {
            return thrown(path, "non-standard exception");
        }
    }

    R require(std::string_view path, Args... args) const {
        if (!function_)
            throw Failure(
                v2::make_error(v2::ConfigErrorCode::invalid_state, std::string(path), "missing callback"));
        try {
            return function_(std::forward<Args>(args)...);
        } catch (const std::exception& error) {
            throw Failure(thrown(path, error.what()), std::current_exception());
        } catch (...) {
            throw Failure(thrown(path, "non-standard exception"), std::current_exception());
        }
    }

  private:
    std::function<R(Args...)> function_;
};

template <class... Args> class Guarded<void(Args...)> final {
  public:
    Guarded() = default;
    template <class F>
        requires(std::is_invocable_r_v<void, F&, Args...> && !std::is_same_v<std::remove_cvref_t<F>, Guarded>)
    Guarded(F&& function) : function_(std::forward<F>(function)) {}

    explicit operator bool() const noexcept {
        return bool(function_);
    }

    v2::Status invoke(std::string_view path, Args... args) const {
        if (!function_)
            return v2::make_error(v2::ConfigErrorCode::invalid_state, std::string(path), "missing callback");
        try {
            function_(std::forward<Args>(args)...);
            return std::nullopt;
        } catch (const std::exception& error) {
            return thrown(path, error.what());
        } catch (...) {
            return thrown(path, "non-standard exception");
        }
    }

    void require(std::string_view path, Args... args) const {
        if (auto error = invoke(path, std::forward<Args>(args)...))
            throw Failure(std::move(*error));
    }

  private:
    std::function<void(Args...)> function_;
};

} // namespace bc::serving::callback
