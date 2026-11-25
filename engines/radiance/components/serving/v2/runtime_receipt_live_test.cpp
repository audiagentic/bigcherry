// Card-free end-to-end receipt capture from a genuine POSIX worker transport.
// The inactive v2 tool must bind to this run, manifest and segment inode.
#include "v2/runtime_receipt.h"
#include "jcs.h"
#include "../../moe-remote/moe_remote_shm.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
namespace shm = r3::moe_shm;
using Json = nlohmann::json;

static int capture(const std::string& executable, const std::string& manifest,
                   const std::string& run, const std::string& dev,
                   const std::string& ino, const std::string& digest,
                   const std::string& directory) {
    const pid_t child = ::fork();
    assert(child >= 0);
    if (child == 0) {
        ::execl(executable.c_str(), executable.c_str(),
                "--manifest", manifest.c_str(),
                "--run-id", run.c_str(),
                "--expected-dev", dev.c_str(),
                "--expected-ino", ino.c_str(),
                "--expected-manifest-digest", digest.c_str(),
                "--record-dir", directory.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    assert(::waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status));
    return WEXITSTATUS(status);
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const auto root = fs::temp_directory_path() /
                      ("bc-receipts-live-" + std::to_string(::getpid()));
    assert(fs::create_directory(root));
    const auto manifest_path = root / "manifest.json";
    const std::string shm_name = "/bc-receipts-live-" + std::to_string(::getpid());
    shm::Layout layout{};
    const shm::LayoutShape shape{16, 1, 2, 16, 8, 16};
    assert(shm::v26::layout_for(shape, 1, shm::v26::Topology{1, 1}, &layout) == shm::Refusal::None);
    const int fd = ::shm_open(shm_name.c_str(),
                              O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600);
    assert(fd >= 0);
    assert(::ftruncate(fd, static_cast<off_t>(layout.bytes)) == 0);
    void* map = ::mmap(nullptr, layout.bytes, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    assert(map != MAP_FAILED);
    struct stat observation{};
    assert(::fstat(fd, &observation) == 0);
    const std::string dev = std::to_string(static_cast<uint64_t>(observation.st_dev));
    const std::string ino = std::to_string(static_cast<uint64_t>(observation.st_ino));

    std::ifstream template_file(std::string(BC_CONTRACT_V2_DOCS_DIR) +
                                "/example.flashnext-4gpu.json");
    assert(template_file);
    Json manifest = Json::parse(template_file);
    manifest["spec"]["handoff"]["transport"]["shm"] = shm_name;
    std::string digest, why;
    assert(bc::serving::manifest_content_digest(manifest, &digest, &why));
    manifest["metadata"]["content_digest"] = digest;
    assert(shm::v26::init(map, layout.bytes, shape, 1, shm::v26::Topology{1, 1},
                          shm::v26::segment_identity(shm_name, false)) == shm::Refusal::None);
    auto* header = static_cast<shm::Header*>(map);
    header->helper_count = 1;
    auto* control = shm::placement_control(map);
    auto* generation = shm::generation_header(map, 0);
    auto* journal = shm::placement_journal(map);
    control->seq = 2;
    control->transaction_id = 0;   // the boot placement: no transaction has begun, the journal is empty
    control->active_epoch = 7;
    control->journal_index = 0;
    generation->epoch = 7;
    generation->ready_helpers = 1;
    assert(shm::valid_self(header, layout.bytes));
    { std::ofstream output(manifest_path);
      assert(output);
      output << manifest.dump();
    }
    {
        const std::string boot_directory = root.string();
        assert(capture(argv[1], manifest_path.string(), "run-boot", dev, ino, digest, boot_directory) == 0);
        std::ifstream boot_saved(root / "run-boot.runtime.json");
        assert(boot_saved);
        const Json boot = Json::parse(boot_saved);
        assert(boot.at("runtime_status") == "active");
        assert(boot.at("snapshots").at(0).at("transaction_id") == 0);
    }
    control->transaction_id = 42;
    journal->entry[0].transaction_id = 42;
    journal->entry[0].before_epoch = 7;
    journal->entry[0].before_digest = control->active_digest;
    assert(shm::valid_self(header, layout.bytes));

    { std::ofstream output(manifest_path);
      assert(output);
      output << manifest.dump();
    }

    const std::string directory = root.string();
    const std::string path = manifest_path.string();
    assert(capture(argv[1], path, "run-001", dev, ino, digest, directory) == 0);
    std::ifstream saved(root / "run-001.runtime.json");
    assert(saved);
    const Json receipt = Json::parse(saved);
    assert(receipt.at("run_id") == "run-001");
    assert(receipt.at("manifest").at("content_digest") == digest);
    assert(receipt.at("runtime_status") == "active");
    assert(receipt.at("snapshots").at(0).at("transaction_id") == 42);
    // An existing receipt is immutable.
    assert(capture(argv[1], path, "run-001", dev, ino, digest, directory) != 0);
    // A new run must not accept an old launch's segment or manifest identity.
    assert(capture(argv[1], path, "run-dev", std::to_string(
                       static_cast<uint64_t>(observation.st_dev) + 1),
                   ino, digest, directory) != 0);
    assert(!fs::exists(root / "run-dev.runtime.json"));
    assert(capture(argv[1], path, "run-manifest", dev, ino,
                   "sha256:" + std::string(64, 'f'), directory) != 0);
    assert(!fs::exists(root / "run-manifest.runtime.json"));
    // An alternate journal from a prior transaction cannot masquerade as live.
    control->journal_index = 1;
    journal->entry[1].transaction_id = 41;
    assert(capture(argv[1], path, "run-stale", dev, ino, digest, directory) != 0);
    assert(!fs::exists(root / "run-stale.runtime.json"));

    assert(::munmap(map, layout.bytes) == 0);
    assert(::close(fd) == 0);
    assert(::shm_unlink(shm_name.c_str()) == 0);
    fs::remove_all(root);
    return 0;
}
