#include "release_archive.h"
#include "release_install.h"
#include "sha256.h"

#include <zlib.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
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

    InstallRequest make(const std::string& tag, Fault fault = Fault::none) {
        const auto folder = home_ / "assets" / tag;
        fs::create_directories(folder);
        const std::string plan(32, 'a');
        const auto archive = folder / ("bigcherryrad-" + plan + ".tar.gz");
        const auto record = folder / "release.receipt.json";

        std::map<std::string, Member> members;
        auto add = [&](const std::string& name, const std::string& data, unsigned mode = 0644) {
            members[name] = Member{name, data, mode, '0'};
        };
        add("bin/radiance", "radiance-binary", 0755);
        add("radiance_home/bin/bc-serve", "serve-binary", 0755);
        add("radiance_home/bin/bc-release", "release-binary", 0755);
        add("radiance_home/helpers/r3_moe_worker", "worker-binary", 0755);
        add("radiance_home/kernels/libr3moe.so", "kernel");
        add("radiance_home/architectures/qwen.so", "architecture");
        add("radiance_home/quantizers/quant.so", "quant");
        add("bigcherry-build-metadata-radiance.json",
            fault == Fault::interpreter_reference ? R"({"interpreter": "python3"})" : R"({"build": "native"})");
        add("configs/flashnext.toml", "schema = 1\n");
        add("configs/qwen27b.toml", "schema = 1\n");

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
            {"binary_sha256", bc::serving::sha256_hex("radiance-binary")},
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
            if (fault == Fault::entry_mode && name == "configs/qwen27b.toml")
                entry.mode = 0755; // disagrees with manifest (not a chmod-based test)
            tar_member(zip, entry);
        }
        if (fault == Fault::duplicate)
            tar_member(zip, members.at("configs/flashnext.toml"));
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
            {"binary_sha256", bc::serving::sha256_hex("radiance-binary")},
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
    check(fs::is_regular_file(a.root / "releases" / a.tag / "configs/flashnext.toml"),
          "installed asset missing self-contained RSC12 config");
    check(fs::is_regular_file(a.root / "releases" / a.tag / "radiance_home/bin/bc-release"),
          "installed asset missing native release tool");
    bc::release::install(b);
    auto second = bc::release::status(a.root, a.sdk);
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
    write_file(a.root / "releases" / a.tag / "configs/qwen27b.toml", "tampered config");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::rollback(a.root, a.sdk); });
    const auto index = bc::release::read_json(a.root / "index.json");
    check(index["current"] == b.tag && index["previous"] == a.tag, "corrupt previous displaced active engine");
    refused(ErrorCode::digest_mismatch, [&] { bc::release::verify(a.root, a.sdk); });
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

void test_real_platform_packer_golden() {
    // A byte-for-byte archive emitted by runtime_assets.pack(), not this
    // test's minimal tar writer. It exercises the real producer's PAX/gzip
    // archive, bundle.json and receipt formats through the native verifier.
    ScopedDir d;
    const auto sdk = d.path / "sdk";
    write_file(sdk / ".info/version", "7.2.4\n");
    const auto fixture = fs::path(BC_RELEASE_GOLDEN_DIR);
    const auto tag = std::string("bc-radiance-1170.99.0");
    bc::release::install({
        tag,
        fixture / "release.receipt.json",
        fixture / "bigcherryrad-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tar.gz",
        d.path / "installed",
        sdk,
    });
    check(bc::release::verify(d.path / "installed", sdk)["current"] == tag,
          "actual platform-produced tar did not install with native verifier");
    check(fs::file_size(d.path / "installed" / "releases" / tag / "configs/flashnext.toml") > 100,
          "actual packaged model config was missing");
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

int main() {
    try {
        test_two_installs_and_rollback();
        test_downgrade_refused_before_unpack();
        test_numeric_version_order();
        test_tamper_refuses_rollback_without_switch();
        test_refusal_matrix();
        test_digest_tag_sdk_and_existing_integrity_refuse();
        test_interrupted_staging_refuses_install();
        test_real_platform_packer_golden();
        std::cout << "bc_release_install_test: native asset/rollback refusals passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bc_release_install_test: " << e.what() << '\n';
        return 1;
    }
}
