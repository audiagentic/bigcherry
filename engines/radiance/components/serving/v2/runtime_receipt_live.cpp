// Inactive v2 collector: read the real helper-owned PlacementControl and
// PlacementJournal from its published POSIX segment, then persist exactly one
// run-bound receipt. Never linked to production bc-serve or the v1 worker.
#include "v2/runtime_receipt.h"
#include "v2/strict_json.h"
#include "generated/v2/serving_codec.h"
#include "jcs.h"

#include <cerrno>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace v2 = bc::serving::v2;

namespace {
bool decimal(const char* text, uint64_t* out) {
  const char* last = text + std::strlen(text);
  auto result = std::from_chars(text, last, *out);
  return result.ec == std::errc{} && result.ptr == last && *out != 0;
}
int refuse(const std::string& detail) {
  std::cerr << "bc-v2-runtime-receipt: refused: " << detail << "\n";
  return 2;
}
// An exclusive file names only this run. A second observation cannot silently
// overwrite earlier evidence; the next capture requires an explicit revision.
bool persist(const std::string& directory, const std::string& run_id,
             const nlohmann::json& receipt) {
  const int dir = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (dir < 0) return false;
  const std::string name = run_id + ".runtime.json";
  const int fd = ::openat(dir, name.c_str(),
                          O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) { ::close(dir); return false; }
  const std::string bytes = receipt.dump() + "\n";
  size_t sent = 0;
  bool ok = true;
  while (sent < bytes.size()) {
    const ssize_t n = ::write(fd, bytes.data() + sent, bytes.size() - sent);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { ok = false; break; }
    sent += static_cast<size_t>(n);
  }
  if (ok) ok = ::fsync(fd) == 0;
  if (::close(fd) != 0) ok = false;
  if (ok) ok = ::fsync(dir) == 0;
  if (!ok) (void)::unlinkat(dir, name.c_str(), 0);
  (void)::close(dir);
  return ok;
}
}

int main(int argc, char** argv) {
  if (argc != 13 || std::string(argv[1]) != "--manifest" ||
      std::string(argv[3]) != "--run-id" ||
      std::string(argv[5]) != "--expected-dev" ||
      std::string(argv[7]) != "--expected-ino" ||
      std::string(argv[9]) != "--expected-manifest-digest" ||
      std::string(argv[11]) != "--record-dir")
    return refuse("usage: --manifest V2.json --run-id RUN --expected-dev DEV --expected-ino INO --expected-manifest-digest DIGEST --record-dir DIR");
  const std::string run_id = argv[4];
  uint64_t expected_dev = 0, expected_ino = 0;
  if (!decimal(argv[6], &expected_dev) || !decimal(argv[8], &expected_ino))
    return refuse("launch-bound shared-memory device and inode required");

  std::ifstream file(argv[2], std::ios::binary);
  if (!file) return refuse("cannot read v2 manifest");
  std::ostringstream buffer;
  buffer << file.rdbuf();
  auto parsed = v2::parse_strict_json(buffer.str(), argv[2]);
  if (!parsed) return refuse("invalid strict v2 manifest JSON");
  v2::Manifest manifest{};
  if (v2::decode(parsed.value(), "", &manifest))
    return refuse("manifest is not a valid v2 contract");
  std::string digest, why;
  if (!bc::serving::manifest_content_digest(parsed.value(), &digest, &why) ||
      digest != manifest.metadata.content_digest || digest != argv[10])
    return refuse("manifest content digest does not verify");
  if (!manifest.spec.handoff)
    return refuse("manifest has no worker handoff");
  const std::string shm_name = manifest.spec.handoff->transport.shm;
  if (shm_name.size() < 2 || shm_name[0] != '/' ||
      shm_name.find('/', 1) != std::string::npos)
    return refuse("invalid POSIX worker segment name");

  const int fd = ::shm_open(shm_name.c_str(), O_RDONLY | O_CLOEXEC, 0);
  if (fd < 0) return refuse("worker segment is absent");
  struct stat observed {};
  if (::fstat(fd, &observed) != 0 ||
      static_cast<uint64_t>(observed.st_dev) != expected_dev ||
      static_cast<uint64_t>(observed.st_ino) != expected_ino ||
      observed.st_size <= 0) {
    ::close(fd);
    return refuse("worker segment identity differs from this run's launch binding");
  }
  const size_t bytes = static_cast<size_t>(observed.st_size);
  void* mapping = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (mapping == MAP_FAILED) return refuse("cannot map worker telemetry");
  auto published = v2::publish_live_runtime_receipt(
      run_id, manifest.spec.handoff->placement.live_policy_state,
      digest, mapping, bytes);
  ::munmap(mapping, bytes);
  if (!published) return refuse("stale, mismatched or unavailable worker placement telemetry");
  auto receipt = std::move(published.value());
  receipt["provenance"] = "worker-shm:" + shm_name + ";dev=" +
                          std::to_string(expected_dev) + ";ino=" +
                          std::to_string(expected_ino);
  if (!persist(argv[12], run_id, receipt))
    return refuse("run receipt already exists or cannot be durably persisted");
  return 0;
}
