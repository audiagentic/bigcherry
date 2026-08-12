// Crash-safe same-directory artifact replacement (HI48).
#pragma once

#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include <stdio.h>
#include <string>

struct ggml_hip_atomic_file {
    FILE * file = nullptr;
    std::string target;
    std::string temporary;
};

bool ggml_hip_atomic_begin(const char * target, ggml_hip_atomic_file & output);
bool ggml_hip_atomic_commit(ggml_hip_atomic_file & output);
void ggml_hip_atomic_abort(ggml_hip_atomic_file & output);

#endif
