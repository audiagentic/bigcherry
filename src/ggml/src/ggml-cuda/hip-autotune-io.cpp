#include "hip-autotune-io.h"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include <errno.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <io.h>
#  include <process.h>
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

bool ggml_hip_atomic_begin(const char * target, ggml_hip_atomic_file & output) {
    if (target == nullptr || target[0] == '\0' || output.file != nullptr) return false;
    output.target = target;
#if defined(_WIN32)
    const unsigned long process = (unsigned long) _getpid();
#else
    const unsigned long process = (unsigned long) getpid();
#endif
    output.temporary = output.target + ".tmp." + std::to_string(process);
    output.file = fopen(output.temporary.c_str(), "wb");
    return output.file != nullptr;
}

void ggml_hip_atomic_abort(ggml_hip_atomic_file & output) {
    if (output.file != nullptr) fclose(output.file);
    output.file = nullptr;
    if (!output.temporary.empty()) remove(output.temporary.c_str());
}

bool ggml_hip_atomic_commit(ggml_hip_atomic_file & output) {
    if (output.file == nullptr) return false;
    bool ok = fflush(output.file) == 0;
#if defined(_WIN32)
    if (ok) ok = _commit(_fileno(output.file)) == 0;
#else
    if (ok) ok = fsync(fileno(output.file)) == 0;
#endif
    if (fclose(output.file) != 0) ok = false;
    output.file = nullptr;
    if (!ok) {
        remove(output.temporary.c_str());
        return false;
    }
#if defined(_WIN32)
    ok = MoveFileExA(
        output.temporary.c_str(), output.target.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    ok = rename(output.temporary.c_str(), output.target.c_str()) == 0;
    if (ok) {
        const size_t slash = output.target.find_last_of('/');
        const std::string directory = slash == std::string::npos
            ? "." : output.target.substr(0, slash);
        const int directory_fd = open(directory.c_str(), O_RDONLY);
        if (directory_fd >= 0) {
            if (fsync(directory_fd) != 0) ok = false;
            close(directory_fd);
        }
    }
#endif
    if (!ok) remove(output.temporary.c_str());
    return ok;
}

#endif
