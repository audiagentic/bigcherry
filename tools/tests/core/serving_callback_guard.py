"""Best-effort source tripwire for common unguarded serving callbacks.

This regex/line-based CTest guard recognizes common injection spellings only;
it is NOT an exhaustive inventory of future callable declarations. The real
boundary for current injection sites is the private typed Guarded/GuardedSymbol
storage and guarded backend adapters. A new raw type may evade the scanner.

Known gaps from REVIEW4 (the scan MISSES both, not safety passes):
  1. int callback_impl(int) { throw 17; }
     using Hook = decltype(&callback_impl);
     struct Owner { Hook callback; int invoke(int x) { return callback(x); } };
  2. class Injected { public: virtual int callback(int) = 0; };
     int use(Injected& cb) { return cb.callback(3); }

Private storage is allowed only in callback_boundary.h and its C-ABI
specialisation. Local recursive functions have exact exemptions. Vendored and
generated code is excluded. RSC13 tracks a clang AST inventory.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3] / "engines" / "radiance" / "components" / "serving"
ALLOWED_FUNCTIONS = {
    "callback_boundary.h": [
        "std::function<R(Args...)> function_;",
        "std::function<void(Args...)> function_;",
    ],
    "config_loader.cpp": [
        "std::function<void(const std::string&)> effective =",
    ],
    "contract/registry.cpp": [
        "std::function<void(size_t)> emit =",
        "std::function<void(size_t)> walk =",
        "std::function<void(size_t)> walk =",
    ],
    "serving_test.cpp": [
        "const std::function<void()>& f",
    ],
}
# Test code may hold a table of its own test functions; the guard is for injected production callables.
ALLOWED_RAW_POINTER_LINES = {
    "session_test.cpp": ["static const std::vector<std::vector<void (*)()>> groups"],
}
FORBIDDEN = (
    (re.compile(r"std::(move_only_function|function_ref)\s*<|boost::function\s*<"), "unwrapped callable"),
    (re.compile(r"\(\s*\*\s*[A-Za-z_]*\w*\s*\)\s*\("), "raw function pointer"),
    (re.compile(r"\b(?:std::invoke|std::bind)\s*\("), "unguarded callable dispatch"),
    (re.compile(r"\b(?:auto|Fn|Callable|Callback)\s*(?:&&|&|\*)?\s+"
                r"(?:callback|callable|hook|predicate)\b"), "unwrapped generic callable"),
    (re.compile(r"\bfriend\s+class\s+(SessionPower|Hardware|Records|Sensor)\b"), "friend-access bypass"),
    (re.compile(r"\b(callback_|\.function_)\s*\("), "unguarded private callable"),
    (re.compile(r"\b(between_|readable|layout_check|recorded|runner|wait|lease_|children_exited_)\s*\("),
     "direct injected callable invocation"),
    (re.compile(r"\b(hooks\.(wait|started|ended|selected_segment)|s_\.plan|in\.runner)\s*\("),
     "direct injected member invocation"),
    (re.compile(r"\b(?<!::)(do_inspect|do_execute|do_save|do_load|do_contains|do_measure|do_journal_path|"
                r"do_unresolved_journals|do_reconcile_recovery)\s*\("),
     "direct raw virtual invocation"),
)
# These virtual operations are private to the base classes and inaccessible
# except via their guarded public adapters.
KNOWN_VIRTUAL = {
    "virtual ~Hardware() = default;",
    "virtual v2::Result<CardSnapshot> do_inspect(const PciAddress& pci) = 0;",
    "virtual v2::Status do_execute(const PowerCommand& command) = 0;",
    "virtual ~Records() = default;",
    "virtual v2::Status do_save(const Journal&) = 0;",
    "virtual v2::Status do_save(const Receipt&) = 0;",
    "virtual v2::Result<std::vector<fs::path>> do_unresolved_journals() = 0;",
    "virtual v2::Result<bool> do_contains(std::string_view run_id) = 0;",
    "virtual v2::Result<Journal> do_load(const fs::path&) = 0;",
    "virtual v2::Status do_reconcile_recovery(const Journal&, const Observation& stop) = 0;",
    "virtual fs::path do_journal_path(std::string_view run_id) const = 0;",
    "virtual ~Sensor() = default;",
    "virtual v2::Result<Window> do_measure(const PciAddress&, WindowPhase, size_t samples) = 0;",
}
ALLOWED_RAW_ADAPTER = {
    'return callback::invoke_backend("power.hardware.inspect", [&] { return do_inspect(pci); });',
    'return callback::invoke_backend("power.hardware.execute", [&] { return do_execute(command); });',
    'return callback::invoke_backend("power.records.save_journal", [&] { return do_save(journal); });',
    'return callback::invoke_backend("power.records.save_receipt", [&] { return do_save(receipt); });',
    'return callback::invoke_backend("power.records.unresolved", [&] { return do_unresolved_journals(); });',
    'return callback::invoke_backend("power.records.contains", [&] { return do_contains(run_id); });',
    'return callback::invoke_backend("power.records.load", [&] { return do_load(path); });',
    '[&] { return do_reconcile_recovery(journal, stop); });',
    '[this](std::string_view id) { return do_journal_path(id); })',
    '[&] { return do_measure(pci, phase, samples); });',
}


def findings(relative: str, content: str) -> list[str]:
    errors = []
    # Per-line detection intentionally errs on the side of denying a new
    # injection; no user-editable suppression marker exists.
    valid_functions = ALLOWED_FUNCTIONS.get(relative, [])
    used = []
    for n, line in enumerate(content.splitlines(), 1):
        if line.lstrip().startswith(("//", "/*", "*")):
            continue
        line = line.split("//", 1)[0].strip()
        if relative in {"callback_boundary.h", "callback_boundary_abi.h"} and re.search(r"\bfriend\b", line):
            errors.append(f"{relative}:{n}: friend bypass in callable boundary")
        if "std::function" in line:
            matches = [i for i, allowed in enumerate(valid_functions)
                       if i not in used and allowed in line]
            if matches:
                used.append(matches[0])
            else:
                errors.append(f"{relative}:{n}: injected std::function outside boundary")
        if line.startswith("virtual ") and not (relative == "power_transaction.h" and line in KNOWN_VIRTUAL):
            errors.append(f"{relative}:{n}: new virtual injection outside sealed power interface")
        if relative not in {"callback_boundary.h", "callback_boundary_abi.h"}:
            for expression, reason in FORBIDDEN:
                if reason == "raw function pointer" and any(
                        allowed in line for allowed in ALLOWED_RAW_POINTER_LINES.get(relative, [])):
                    continue
                # Raw virtual calls are legal only in the inline sealed
                # adapters (each of which calls invoke_backend/Guarded.require).
                if reason == "direct raw virtual invocation":
                    if relative == "power_transaction.h" and (
                        line in ALLOWED_RAW_ADAPTER or "virtual " in line or "override" in line
                    ):
                        continue
                    if relative.endswith("_test.cpp") and "override" in line:
                        continue
                    if "::do_" in line:  # a C++ out-of-line override definition
                        continue
                if reason == "direct injected callable invocation" and "std::move(" in line:
                    # A member-initializer is not an invocation of the stored function.
                    continue
                if expression.search(line):
                    errors.append(f"{relative}:{n}: {reason}")
    return errors


def check(root: pathlib.Path) -> list[str]:
    errors = []
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.suffix not in {".h", ".hpp", ".cpp", ".cc"}:
            continue
        relative = source.relative_to(root).as_posix()
        if any(part in {"third_party", "generated"} for part in source.relative_to(root).parts):
            continue
        errors.extend(findings(relative, source.read_text(errors="replace")))
    return errors


def self_test() -> None:
    assert findings("session.h", "std::function<bool()> rogue;")
    assert findings("session.cpp", "if (hooks.started(...)) {}")
    assert findings("power_diagnostics.cpp", "between_();")
    assert findings("new_module.h", "virtual bool injection() = 0;")
    assert findings("new_module.cpp", "bool (*raw_hook)(int);")
    assert findings("new_module.cpp", "using Callback = int (*)(int);")
    assert findings("new_module.cpp", "template<class F> void f(auto callback) { callback(); }")
    assert findings("new_module.cpp", "auto result = std::invoke(callback);")
    assert findings("callback_boundary.h", "friend class SessionPower;")
    assert not findings("config_loader.cpp", "std::function<void(const std::string&)> effective = fn;")


if __name__ == "__main__":
    self_test()
    errors = check(ROOT)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print("callback source tripwire: PASS (recognized patterns only; not exhaustive)")
