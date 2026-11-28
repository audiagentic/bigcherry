#include "release_archive.h"
#include "release_install.h"
#include "sha256.h"

#include <zlib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {
using bc::release::Error;
using bc::release::ErrorCode;
using bc::release::InstallRequest;
using bc::release::Json;
namespace fs = std::filesystem;

void check(bool passed, const std::string& reason) {
    if (!passed)
        throw std::runtime_error(reason);
}

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    check(in.good(), "could not read executable fixture: " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    check(out.good(), "could not create fixture file");
}

struct ScopedDir {
    fs::path path;
    ScopedDir() {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::path(BC_RELEASE_TEST_ROOT) /
               ("bc-release-test-" + std::to_string(::getpid()) + "-" + std::to_string(ticks));
        fs::create_directories(path);
    }
    ~ScopedDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

struct Member {
    std::string name;
    std::string content;
    unsigned mode = 0644;
    char type = '0';
};

void tar_header(gzFile zip, const Member& member) {
    check(member.name.size() < 100, "fixture tar path too long");
    std::array<char, 512> header{};
    std::copy(member.name.begin(), member.name.end(), header.begin());
    auto octal = [&](size_t at, size_t width, uint64_t number) {
        char output[32]{};
        std::snprintf(output, sizeof(output), "%0*llo", static_cast<int>(width - 1),
                      static_cast<unsigned long long>(number));
        check(std::string(output).size() <= width - 1, "tar value overflow in fixture");
        std::copy_n(output, width - 1, header.begin() + static_cast<std::ptrdiff_t>(at));
    };
    octal(100, 8, member.mode);
    octal(108, 8, 0);
    octal(116, 8, 0);
    octal(124, 12, member.content.size());
    octal(136, 12, 0);
    std::fill(header.begin() + 148, header.begin() + 156, ' ');
    header[156] = member.type;
    std::copy_n("ustar", 5, header.begin() + 257);
    std::copy_n("00", 2, header.begin() + 263);
    uint64_t sum = 0;
    for (unsigned char c : header)
        sum += c;
    octal(148, 8, sum);
    check(gzwrite(zip, header.data(), header.size()) == static_cast<int>(header.size()),
          "cannot write tar fixture header");
}

void tar_member(gzFile zip, const Member& member) {
    tar_header(zip, member);
    if (!member.content.empty())
        check(gzwrite(zip, member.content.data(), static_cast<unsigned>(member.content.size())) ==
                  static_cast<int>(member.content.size()),
              "cannot write tar fixture bytes");
    const size_t pad = (512 - (member.content.size() % 512)) % 512;
    std::array<char, 512> zeros{};
    if (pad)
        check(gzwrite(zip, zeros.data(), static_cast<unsigned>(pad)) == static_cast<int>(pad),
              "cannot write tar padding");
}

enum class Fault : std::uint8_t {
    none,
    duplicate,
    traversal,
    symlink,
    extra,
    python_file,
    shebang,
    interpreter_reference,
    native_profile_interpreter,
    native_profile_executable,
    unknown_profile,
    benchmark_catalog,
    moe_benchmark_catalog,
    entry_mode,
    wrong_manifest,
    wrong_source,
};

class Fixture {
  public:
    explicit Fixture(const fs::path& home) : home_(home) {
        fs::create_directories(home_ / "sdk/.info");
        write_file(home_ / "sdk/.info/version", "7.2.4\n");
    }

    InstallRequest make(const std::string& tag, Fault fault = Fault::none, bool executable = false,
                        bool native_profile = false) {
        const auto folder = home_ / "assets" / tag;
        fs::create_directories(folder);
        const std::string plan(32, 'a');
        const auto archive = folder / ("bigcherryrad-" + plan + ".tar.gz");
        const auto record = folder / "release.receipt.json";

        std::map<std::string, Member> members;
        auto add = [&](const std::string& name, const std::string& data, unsigned mode = 0644) {
            members[name] = Member{name, data, mode, '0'};
        };
        const std::string engine_bytes = executable ? read_file(BC_FAKE_ENGINE) : "radiance-binary";
        add("bin/radiance", engine_bytes, 0755);
        add("radiance_home/bin/bc-serve", executable ? read_file(BC_SERVE_EXE) : "serve-binary", 0755);
        add("radiance_home/bin/bc-release", executable ? read_file(BC_RELEASE_EXE) : "release-binary", 0755);
        add("radiance_home/helpers/r3_moe_worker", executable ? engine_bytes : "worker-binary", 0755);
        add("radiance_home/kernels/libr3moe.so", "kernel");
        add("radiance_home/architectures/qwen.so", "architecture");
        add("radiance_home/quantizers/quant.so", "quant");
        std::string metadata = R"({"build": "native"})";
        if (fault == Fault::interpreter_reference) {
            metadata = R"({"interpreter": "python3"})";
        } else if (executable) {
            metadata = Json({{"toolchain", Json::array({Json::array({"rocm_root", (home_ / "sdk").string()}),
                                                        Json::array({"rocm_version", "7.2.4"})})}})
                           .dump();
        }
        add("bigcherry-build-metadata-radiance.json", metadata);
        // A config is its native YAML: the bundle carries one file per named config (the second one here stands for the rest).
        add("configs/flashnext-4gpu.yaml",
            fault == Fault::native_profile_interpreter ? "# requires python3\ncards: [0]\n" : "cards: [0]\n",
            fault == Fault::native_profile_executable ? 0755 : 0644);
        add("configs/qwen27b-tp3-long-q8.yaml", "cards: [0, 1, 2]\n");
        if (fault == Fault::unknown_profile)
            add("configs/unreviewed.yaml", "cards: [0]\n");
        add("heat/flashnext-mixed.hot-first", "mixed production heat seed\n");
        add("heat/perf51.rank0.hot-first", "baseline heat seed\n");

        if (fault == Fault::benchmark_catalog)
            add("radiance_home/kernels.rkb", "benchmark fixture");
        if (fault == Fault::moe_benchmark_catalog)
            add("radiance_home/kernels_moe.rkb", "benchmark fixture");
        if (fault == Fault::python_file)
            add("radiance_home/vectors/config.py", "print('must refuse')\n");
        if (fault == Fault::shebang)
            add("radiance_home/vectors/entry.dat", "#!/usr/bin/env python3\npass\n");

        Json checks = Json::object();
        std::vector<std::string> names;
        for (const auto& [name, member] : members) {
            names.push_back(name);
            checks[name] = {{"sha256", bc::serving::sha256_hex(member.content)},
                            {"size", member.content.size()},
                            {"mode", member.mode}};
        }
        Json manifest = {
            {"schema", "bigcherry-radiance-runtime-v1"},
            {"product", "bigcherryrad"},
            {"payload_scope", "engine-runtime"},
            {"release_tag", tag},
            {"plan_id", plan},
            {"source_slice_id", "fixture-slice"},
            {"binary_sha256", bc::serving::sha256_hex(engine_bytes)},
            {"rocm_version", "7.2.4"},
            {"engine_tree", std::string(40, 'b')},
            {"config_tree", std::string(40, 'c')},
            {"files", checks},
        };
        if (fault == Fault::wrong_source)
            manifest["engine_tree"] = std::string(40, 'f');
        const std::string bytes = manifest.dump() + "\n";
        gzFile zip = gzopen(archive.c_str(), "wb");
        check(zip != nullptr, "could not create fixture gzip");
        tar_member(zip, Member{"bundle.json", bytes});
        for (const auto& [name, member] : members) {
            Member entry = member;
            if (fault == Fault::entry_mode && name == "configs/qwen27b-tp3-long-q8.yaml")
                entry.mode = 0755; // disagrees with manifest (not a chmod-based test)
            tar_member(zip, entry);
        }
        if (fault == Fault::duplicate)
            tar_member(zip, members.at("configs/flashnext-4gpu.yaml"));
        if (fault == Fault::traversal)
            tar_member(zip, Member{"../escape", "malicious"});
        if (fault == Fault::extra)
            tar_member(zip, Member{"bin/unapproved", "malicious"});
        if (fault == Fault::symlink)
            tar_member(zip, Member{"bin/link", "", 0755, '2'});
        std::array<char, 1024> zeros{};
        check(gzwrite(zip, zeros.data(), zeros.size()) == static_cast<int>(zeros.size()),
              "could not finish fixture tar");
        check(gzclose(zip) == Z_OK, "could not close fixture archive");

        const auto manifest_sha =
            fault == Fault::wrong_manifest ? std::string(64, '0') : bc::serving::sha256_hex(bytes);
        const Json receipt = {
            {"schema", "bigcherry-radiance-runtime-v1"},
            {"product", "bigcherryrad"},
            {"payload_scope", "engine-runtime"},
            {"release_tag", tag},
            {"name", archive.filename().string()},
            {"checksum_name", archive.filename().string() + ".sha256"},
            {"sha256", bc::release::sha256_file(archive)},
            {"manifest_sha256", manifest_sha},
            {"file_names", names},
            {"plan_id", plan},
            {"source_slice_id", "fixture-slice"},
            {"binary_sha256", bc::serving::sha256_hex(engine_bytes)},
            {"rocm_version", "7.2.4"},
            {"engine_tree", std::string(40, 'b')},
            {"config_tree", std::string(40, 'c')},
        };
        write_file(record, receipt.dump(2));
        return {tag, record, archive, home_ / "installed", home_ / "sdk"};
    }

    const fs::path& home() const {
        return home_;
    }

  private:
    fs::path home_;
};

template <typename Operation> void refused(ErrorCode expected, Operation&& operation) {
    try {
        operation();
    } catch (const Error& e) {
        check(e.code() == expected, std::string("unexpected native error kind: ") + e.what());
        return;
    }
    throw std::runtime_error("unsafe release operation unexpectedly succeeded");
}

void test_two_installs_and_rollback() {
    ScopedDir d;
    Fixture f(d.path);
    auto a = f.make("bc-radiance-1170.31.0");
    auto b = f.make("bc-radiance-1170.32.0");
    bc::release::install(a);
    auto first = bc::release::verify(a.root, a.sdk);
    check(first["current"] == a.tag && first["previous"].is_null(), "first verified release not current");
    check(fs::is_regular_file(a.root / "releases" / a.tag / "configs/flashnext-4gpu.yaml"),
          "installed asset missing self-contained RSC12 config");
    check(fs::is_regular_file(a.root / "releases" / a.tag / "heat/flashnext-mixed.hot-first") &&
          fs::is_regular_file(a.root / "releases" / a.tag / "heat/perf51.rank0.hot-first"),
          "installed asset missing verified heat seed closure");
    check(fs::is_regular_file(a.root / "releases" / a.tag / "radiance_home/bin/bc-release"),
          "installed asset missing native release tool");
    bc::release::install(b);
    auto second = bc::release::verify(a.root, a.sdk);
    check(second["current"] == b.tag && second["previous"] == a.tag,
          "two-version current/previous index incorrect");
    check(bc::release::rollback(a.root, a.sdk)["current"] == a.tag,
          "rollback did not reverify previous release");
    check(bc::release::verify(a.root, a.sdk)["previous"] == b.tag,
          "rollback did not keep the displaced current");
    // Reinstall the former current without allocating a third, including
    // rechecking that its tag-bound release receipt is identical.
    bc::release::install(b);
    check(bc::release::verify(a.root, a.sdk)["current"] == b.tag,
          "verified reinstall did not select requested exact tag");
}

// Execute the actual native CLI without a shell or a Python launcher.
int cli_install(const InstallRequest& request, const char* authorization) {
    const pid_t pid = ::fork();
    check(pid >= 0, "fork for native CLI failed");
    if (pid == 0) {
        ::execl(BC_RELEASE_BINARY_PATH, BC_RELEASE_BINARY_PATH, "install", "--tag", request.tag.c_str(),
                "--record", request.record.c_str(), "--archive", request.archive.c_str(), "--root",
                request.root.c_str(), "--sdk", request.sdk.c_str(), "--allow-downgrade", authorization,
                static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status = 0;
    check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status), "native CLI terminated abnormally");
    return WEXITSTATUS(status);
}

void test_downgrade_refused_before_unpack() {
    ScopedDir d;
    Fixture f(d.path);
    const auto a = f.make("bc-radiance-1170.31.0");
    const auto b = f.make("bc-radiance-1170.32.0");
    const auto old = f.make("bc-radiance-1170.30.0");
    bc::release::install(a);
    bc::release::install(b);
    refused(ErrorCode::conflict, [&] { bc::release::install(old); });
    const auto still = bc::release::verify(a.root, a.sdk);
    check(still["current"] == b.tag && still["previous"] == a.tag,
          "unflagged downgrade changed installed current/previous");
    check(!fs::exists(a.root / "releases" / old.tag),
          "unflagged downgrade should refuse before extracting the older release");
    check(cli_install(old, "yes") == static_cast<int>(ErrorCode::invalid_input),
          "CLI accepted untyped downgrade authorization");
    check(cli_install(old, "false") == static_cast<int>(ErrorCode::conflict),
          "CLI accepted explicitly denied downgrade");
    check(cli_install(old, "true") == 0, "CLI rejected explicitly authorized downgrade");
    const auto downgraded = bc::release::verify(a.root, a.sdk);
    check(downgraded["current"] == old.tag && downgraded["previous"] == b.tag,
          "authorized downgrade did not retain previous exact release");
    check(bc::release::rollback(a.root, a.sdk)["current"] == b.tag,
          "explicit rollback cannot restore the verified newer release");
    const auto numeric_alias = f.make("bc-radiance-1170.032.0");
    refused(ErrorCode::conflict, [&] { bc::release::install(numeric_alias); });
    auto authorized_alias = numeric_alias;
    authorized_alias.allow_downgrade = true;
    bc::release::install(authorized_alias);
    check(bc::release::verify(a.root, a.sdk)["previous"] == b.tag,
          "equal numeric version with different tag did not retain prior current");
}

void test_numeric_version_order() {
    ScopedDir d;
    Fixture f(d.path);
    auto nine = f.make("bc-radiance-1170.9.0");
    auto ten = f.make("bc-radiance-1170.10.0");
    bc::release::install(nine);
    bc::release::install(ten);
    check(bc::release::verify(nine.root, nine.sdk)["current"] == ten.tag,
          "integer version order incorrectly used lexicographic comparison");
}

void test_tamper_refuses_rollback_without_switch() {
    ScopedDir d;
    Fixture f(d.path);
    const auto a = f.make("bc-radiance-1170.33.0");
    const auto b = f.make("bc-radiance-1170.34.0");
    bc::release::install(a);
    bc::release::install(b);
    write_file(a.root / "releases" / a.tag / "configs/qwen27b-tp3-long-q8.yaml", "tampered config");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::rollback(a.root, a.sdk); });
    const auto index = bc::release::read_json(a.root / "index.json");
    check(index["current"] == b.tag && index["previous"] == a.tag, "corrupt previous displaced active engine");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify(a.root, a.sdk); });
    // A different installed release's heat seed is as immutable as its
    // engine and config: digest tampering must make verification fail closed.
    write_file(b.root / "releases" / b.tag / "heat/flashnext-mixed.hot-first", "changed heat");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify(b.root, b.sdk); });
}

void test_refusal_matrix() {
    const std::vector<std::pair<Fault, ErrorCode>> faults = {
        {Fault::duplicate, ErrorCode::unsafe_archive},
        {Fault::traversal, ErrorCode::unsafe_archive},
        {Fault::symlink, ErrorCode::unsafe_archive},
        {Fault::extra, ErrorCode::unsafe_archive},
        {Fault::python_file, ErrorCode::unsafe_archive},
        {Fault::shebang, ErrorCode::unsafe_archive},
        {Fault::interpreter_reference, ErrorCode::unsafe_archive},
        {Fault::native_profile_interpreter, ErrorCode::unsafe_archive},
        {Fault::native_profile_executable, ErrorCode::unsafe_archive},
        {Fault::unknown_profile, ErrorCode::unsafe_archive},
        {Fault::entry_mode, ErrorCode::digest_mismatch},
        {Fault::wrong_manifest, ErrorCode::digest_mismatch},
        {Fault::wrong_source, ErrorCode::wrong_identity},
    };
    for (size_t i = 0; i < faults.size(); ++i) {
        ScopedDir d;
        Fixture f(d.path);
        const auto tag = "bc-radiance-1170." + std::to_string(40 + i) + ".0";
        const auto request = f.make(tag, faults[i].first);
        refused(faults[i].second, [&] { bc::release::install(request); });
        check(!fs::exists(request.root / "index.json"), "invalid archive published active release index");
    }
}

void test_optional_native_profile() {
    ScopedDir d;
    Fixture f(d.path);
    const auto request = f.make("bc-radiance-1170.52.0", Fault::none, false, true);
    bc::release::install(request);
    const auto profile = request.root / "releases" / request.tag / "configs/flashnext-4gpu.yaml";
    check(read_file(profile) == "cards: [0]\n", "native profile bytes changed during installation");
    check(read_file(request.root / "releases" / request.tag / "configs/qwen27b-tp3-long-q8.yaml") ==
              "cards: [0, 1, 2]\n", "a second native profile was not installed");
    check(bc::release::verify(request.root, request.sdk)["current"] == request.tag,
          "native profile bundle failed verification");
    write_file(profile, "cards: [1]\n");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify(request.root, request.sdk); });
}

void test_digest_tag_sdk_and_existing_integrity_refuse() {
    ScopedDir d;
    Fixture f(d.path);
    auto a = f.make("bc-radiance-1170.60.0");
    auto b = f.make("bc-radiance-1170.61.0");
    auto wrong_tag = b;
    wrong_tag.tag = "bc-radiance-1170.99.0";
    refused(ErrorCode::wrong_identity, [&] { bc::release::install(wrong_tag); });
    auto malformed = b;
    malformed.tag = "../escape";
    refused(ErrorCode::invalid_input, [&] { bc::release::install(malformed); });
    auto wrong_sdk = b;
    const auto other_sdk = d.path / "sdk-mismatch";
    write_file(other_sdk / ".info/version", "7.1.0");
    wrong_sdk.sdk = other_sdk;
    refused(ErrorCode::sdk_mismatch, [&] { bc::release::install(wrong_sdk); });
    bc::release::install(a);
    {
        std::ofstream out(b.archive, std::ios::app | std::ios::binary);
        out << "modified archive bytes";
    }
    refused(ErrorCode::digest_mismatch, [&] { bc::release::install(b); });
    check(bc::release::verify(a.root, a.sdk)["current"] == a.tag,
          "wrong-tag/digest/Sdk attempt switched the verified engine");
    write_file(a.root / "releases" / a.tag / "install.json", "{}");
    refused(ErrorCode::invalid_install, [&] { bc::release::verify(a.root, a.sdk); });
    refused(ErrorCode::invalid_install, [&] { bc::release::install(a); });
}

void test_legacy_release_profile_name_refused() {
    // This published-release fixture retains the retired profile name in its
    // signed manifest. Strict current-name installation must reject it; do not
    // rewrite the historical receipt/archive or reintroduce an old-name alias.
    ScopedDir d;
    const auto sdk = d.path / "sdk";
    write_file(sdk / ".info/version", "7.2.4\n");
    const auto fixture = fs::path(BC_RELEASE_GOLDEN_DIR);
    const auto tag = std::string("bc-radiance-1170.99.0");
    refused(ErrorCode::unsafe_archive, [&] {
        bc::release::install({
            tag,
            fixture / "release.receipt.json",
            fixture / "bigcherryrad-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tar.gz",
            d.path / "installed",
            sdk,
        });
    });
}

// Exercise the shipped CLI entrypoints, not just release-core functions. The
// real bc-serve is installed in the freshly built asset with the fake GPU engine.
int run_cli(const fs::path& exe, const std::vector<std::string>& args, const fs::path& output,
            const fs::path& installed_engine = {}) {
    const pid_t pid = ::fork();
    check(pid >= 0, "fork of native release CLI failed");
    if (pid == 0) {
        const int fd = ::open(output.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
        if (fd < 0 || ::dup2(fd, STDOUT_FILENO) < 0 || ::dup2(fd, STDERR_FILENO) < 0)
            ::_exit(127);
        ::close(fd);
        ::unsetenv("BIGCHERRY_CONFIGS");
        // The isolated temporary SDK is intentionally not the CI host's SDK.
        // Do not inherit the host's ROCm override into a synthetic release.
        ::unsetenv("ROCM_PATH");
        ::unsetenv("HIP_PATH");
        // The same guard refuses an LD_LIBRARY_PATH entry that carries ROCm libraries outside the recorded SDK (a build
        // container exports its own SDK's lib directory there), so the host's search path is not inherited either.
        ::unsetenv("LD_LIBRARY_PATH");
        std::vector<std::string> owned = {exe.string()};
        owned.insert(owned.end(), args.begin(), args.end());
        if (!installed_engine.empty()) {
            owned.push_back("--installed-engine");
            owned.push_back(installed_engine.string());
        }
        std::vector<char*> argv;
        argv.reserve(owned.size() + 1);
        for (auto& text : owned)
            argv.push_back(text.data());
        argv.push_back(nullptr);
        ::execv(exe.c_str(), argv.data());
        ::_exit(127);
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(240);
    int status = 0;
    for (;;) {
        const auto ended = ::waitpid(pid, &status, WNOHANG);
        if (ended == pid)
            return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
        check(ended >= 0, "waitpid failed");
        if (std::chrono::steady_clock::now() >= deadline) {
            ::kill(pid, SIGKILL);
            (void)::waitpid(pid, &status, 0);
            throw std::runtime_error("native CLI did not return within test deadline");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void test_native_installed_start_gate() {
    ScopedDir d;
    Fixture f(d.path);
    auto a = f.make("bc-radiance-1170.70.0", Fault::none, true);
    const auto installed = a.root / "releases" / a.tag;
    const auto cli = fs::path(BC_RELEASE_EXE);
    const auto log = d.path / "cli.log";
    const int installed_rc = run_cli(cli,
                                     {"install", "--tag", a.tag, "--record", a.record.string(), "--archive",
                                      a.archive.string(), "--root", a.root.string(), "--sdk", a.sdk.string()},
                                     log);
    check(installed_rc == 0,
          "bc-release CLI failed to install freshly packaged native executables: " + read_file(log));
    const auto serve = installed / "radiance_home/bin/bc-serve";
    const auto run = d.path / "run";
    // The same installed asset also powers the existing real detached owner
    // and strict v2 replay test. Its fake process is drawn from PREFIX/bin,
    // not the component test build: this is a card-free release->start gate.
    const int replay = run_cli(BC_SESSION_STORED_TEST, {}, log, installed / "bin/radiance");
    check(replay == 0,
          "native stored-manifest session did not start the installed fake engine: " + read_file(log));
    // A verified prefix must select its packaged RSC12 configs (rather
    // than treating the name as a legacy layouts entry). The minimal test
    // package intentionally has no named configs, so resolution refuses
    // with a named-config error after the installed-prefix gate.
    const int valid = run_cli(serve,
                              {"serve", "qwen27b-tp3-q8", "--run-dir", run.string(), "--site",
                               (d.path / "missing-site.toml").string()},
                              log);
    check(valid != 0 && read_file(log).find("release refusal") == std::string::npos &&
              read_file(log).find("sdk-check") == std::string::npos,
          "verified prefix should reach real resolver: " + read_file(log));

    // The executable must not verify one asset then launch another engine
    // or source plugins from an unverified runtime home.
    for (const std::vector<std::string> bypass :
         {std::vector<std::string>{"--engine-binary", "/bin/true"},
          std::vector<std::string>{"--set", "engine.radiance_home=\"/tmp/not-the-install\""}}) {
        std::vector<std::string> args = {"serve", "qwen27b-tp3-q8", "--run-dir", run.string()};
        args.insert(args.end(), bypass.begin(), bypass.end());
        const int rc = run_cli(serve, args, log);
        check(rc != 0 && read_file(log).find("installed release refusal") != std::string::npos,
              "installed release allowed substituting a different runtime: " + read_file(log));
    }

    // Missing receipt is not equivalent to a checkout; both entrypoints
    // refuse before they ever plan or launch a child.
    const std::string original_install_receipt = read_file(installed / "install.json");
    fs::remove(installed / "install.json");
    for (const auto& mode : {"serve", "start"}) {
        const int rc = run_cli(serve, {mode, "qwen27b-tp3-q8", "--run-dir", run.string()}, log);
        check(rc != 0 && read_file(log).find("installed release refusal") != std::string::npos,
              std::string(mode) + " did not refuse the missing receipt: " + read_file(log));
    }
    // Reinstall and flip the current tag. The old still-retained release
    // is valid for rollback, never for a new serve/start.
    write_file(installed / "install.json", original_install_receipt);
    check(bc::release::verify(a.root, a.sdk)["current"] == a.tag, "restored fixture is not verified");
    const auto b = f.make("bc-radiance-1170.71.0", Fault::none, true);
    bc::release::install(b);
    const int retired = run_cli(serve, {"start", "qwen27b-tp3-q8", "--run-dir", run.string()}, log);
    check(retired != 0 && read_file(log).find("not the verified current tag") != std::string::npos,
          "retained old prefix started despite current tag change: " + read_file(log));
    const auto current_serve = b.root / "releases" / b.tag / "radiance_home/bin/bc-serve";
    write_file(b.root / "releases" / b.tag / "configs/qwen27b-tp3-long-q8.yaml", "cards: [9]\n");
    const int tampered = run_cli(current_serve, {"serve", "qwen27b-tp3-q8", "--run-dir", run.string()}, log);
    check(tampered != 0 && read_file(log).find("BC_RELEASE_3") != std::string::npos,
          "tampered current release reached serving: " + read_file(log));
}

void test_benchmark_catalogs_are_not_native_runtime_members() {
    ScopedDir d;
    Fixture f(d.path);
    for (const auto fault : {Fault::benchmark_catalog, Fault::moe_benchmark_catalog}) {
        auto req = f.make(fault == Fault::benchmark_catalog ? "bc-radiance-1170.64.0" :
                                                              "bc-radiance-1170.65.0",
                          fault);
        refused(ErrorCode::unsafe_archive, [&] { bc::release::install(req); });
    }
}

void test_interrupted_staging_refuses_install() {
    ScopedDir d;
    Fixture f(d.path);
    auto a = f.make("bc-radiance-1170.63.0");
    fs::create_directories(a.root / ".staging-interrupted");
    write_file(a.root / ".staging-interrupted/incomplete", "not published");
    refused(ErrorCode::invalid_install, [&] { bc::release::install(a); });
    check(!fs::exists(a.root / "index.json"), "staging residue incorrectly became the current engine");
}

} // namespace

// A start verifies the current install by the stamps recorded when it was verified in full (install, `verify`): an
// unchanged install hashes no runtime file; anything that moved is hashed and judged by the existing full check.
// Let the file system's clock tick, so a change after this call moves a file's stamps.
void tick() { ::usleep(20000); }

void test_start_verification_uses_stamps() {
    ScopedDir d;
    Fixture f(d.path);
    const auto a = f.make("bc-radiance-1170.41.0");
    const auto b = f.make("bc-radiance-1170.42.0");
    bc::release::install(a);
    bc::release::install(b);
    const auto prefix = b.root / "releases" / b.tag;
    const auto member = prefix / "radiance_home/kernels/libr3moe.so";

    tick();
    (void)bc::release::verify(b.root, b.sdk);   // a full check: the record a start trusts
    uint64_t hashed = bc::release::hashed_files();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag, "start verification of the current tag");
    check(bc::release::hashed_files() == hashed, "a start on an unchanged install hashes no runtime file");

    // The explicit verify is still the full hash, of the current tag and the previous one.
    tick();
    check(bc::release::verify(b.root, b.sdk)["current"] == b.tag, "explicit verify");
    check(bc::release::hashed_files() > hashed, "explicit verify hashes every member");
    hashed = bc::release::hashed_files();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() == hashed,
          "a start after an explicit verify still hashes nothing");

    // The previous tag is not scanned on a start: tampering with it does not refuse the start, but still refuses verify.
    write_file(a.root / "releases" / a.tag / "radiance_home/kernels/libr3moe.so", "tampered previous");
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() == hashed,
          "a start does not scan the previous tag");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify(b.root, b.sdk); });

    // Each kind of change to the current install is caught: same-size content, an added member, a removed member.
    write_file(member, "kernel");
    tick();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag, "restored bytes pass after a re-hash");
    check(bc::release::hashed_files() > hashed, "rewritten (even identical) bytes are hashed again");
    hashed = bc::release::hashed_files();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() == hashed,
          "the re-hash re-stored the stamps");
    write_file(member, "kernel!");   // different content
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify_current(b.root, b.sdk); });
    write_file(member, "kernel");
    tick();
    (void)bc::release::verify_current(b.root, b.sdk);
    write_file(prefix / "radiance_home/kernels/extra.so", "added");
    refused(ErrorCode::invalid_install, [&] { bc::release::verify_current(b.root, b.sdk); });
    fs::remove(prefix / "radiance_home/kernels/extra.so");
    tick();
    (void)bc::release::verify_current(b.root, b.sdk);
    fs::remove(member);
    refused(ErrorCode::invalid_install, [&] { bc::release::verify_current(b.root, b.sdk); });
    write_file(member, "kernel");
    tick();
    (void)bc::release::verify_current(b.root, b.sdk);

    // A touch moves the change time: the member is hashed (and passes), and the stamps are stored again.
    hashed = bc::release::hashed_files();
    const timespec when[2] = {{1000000000, 0}, {1000000000, 0}};
    tick();
    check(::utimensat(AT_FDCWD, member.c_str(), when, 0) == 0, "touch");
    tick();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag, "a touched member passes");
    check(bc::release::hashed_files() > hashed, "a touched member is hashed, not trusted");
    hashed = bc::release::hashed_files();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() == hashed,
          "and then costs nothing again");

    // A corrupt or missing stamp file is ignored: the full hash runs.
    tick();
    write_file(b.root / "stamps" / (b.tag + ".json"), "{ not json");
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() > hashed,
          "a corrupt stamp file is ignored");
    fs::remove(b.root / "stamps" / (b.tag + ".json"));
    hashed = bc::release::hashed_files();
    check(bc::release::verify_current(b.root, b.sdk)["current"] == b.tag && bc::release::hashed_files() > hashed,
          "a missing stamp file is ignored");
}

int main() {
    try {
        test_start_verification_uses_stamps();
        test_two_installs_and_rollback();
        test_downgrade_refused_before_unpack();
        test_numeric_version_order();
        test_tamper_refuses_rollback_without_switch();
        test_refusal_matrix();
        test_optional_native_profile();
        test_digest_tag_sdk_and_existing_integrity_refuse();
        test_benchmark_catalogs_are_not_native_runtime_members();
        test_interrupted_staging_refuses_install();
        test_legacy_release_profile_name_refused();
        test_native_installed_start_gate();
        std::cout << "bc_release_install_test: native asset/rollback refusals passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bc_release_install_test: " << e.what() << '\n';
        return 1;
    }
}
