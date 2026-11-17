#include "release_install.h"

#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
using bc::release::Error;
using bc::release::ErrorCode;

[[noreturn]] void usage(const std::string& error) {
    throw Error(ErrorCode::invalid_input,
                error +
                    "\nusage: bc-release install --tag TAG --record FILE --archive FILE --root DIR --sdk DIR"
                    "\n       bc-release verify|status|rollback --root DIR --sdk DIR");
}

std::map<std::string, std::string> options(int argc, const char* const argv[]) {
    std::map<std::string, std::string> opts;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || std::string_view(argv[i]).substr(0, 2) != "--")
            usage("options require --name VALUE pairs");
        const auto key = std::string(argv[i]).substr(2);
        if ((key != "root" && key != "sdk" && key != "tag" && key != "record" && key != "archive") ||
            !opts.emplace(key, argv[i + 1]).second || opts[key].empty())
            usage("unknown, empty or repeated option: " + key);
    }
    return opts;
}

const std::string& needed(const std::map<std::string, std::string>& args, const char* name) {
    const auto it = args.find(name);
    if (it == args.end())
        usage(std::string("required option missing: --") + name);
    return it->second;
}

} // namespace

int main(int argc, const char* const argv[]) {
    try {
        if (argc < 2)
            usage("missing operation");
        const std::string command = argv[1];
        const auto args = options(argc, argv);
        const auto root = needed(args, "root");
        const auto sdk = needed(args, "sdk");
        if (command == "install") {
            if (args.size() != 5)
                usage("install needs only --tag, --record, --archive, --root, --sdk");
            bc::release::install(
                {needed(args, "tag"), needed(args, "record"), needed(args, "archive"), root, sdk});
            std::cout << bc::release::status(root, sdk).dump(2) << '\n';
        } else if (command == "verify" || command == "status" || command == "rollback") {
            if (args.size() != 2)
                usage("verify/status/rollback accept only --root and --sdk");
            bc::release::Json state;
            if (command == "rollback")
                state = bc::release::rollback(root, sdk);
            else if (command == "verify")
                state = bc::release::verify(root, sdk);
            else
                state = bc::release::status(root, sdk);
            std::cout << state.dump(2) << '\n';
        } else {
            usage("unknown bc-release operation: " + command);
        }
        return 0;
    } catch (const Error& error) {
        std::cerr << "BC_RELEASE_" << static_cast<int>(error.code()) << ": " << error.what() << '\n';
        return static_cast<int>(error.code());
    } catch (const std::exception& error) {
        std::cerr << "BC_RELEASE_IO: " << error.what() << '\n';
        return static_cast<int>(ErrorCode::io);
    }
}
