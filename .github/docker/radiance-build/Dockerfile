# Build image for the radiance CI job (.github/workflows/radiance-build.yml): everything the pinned radiance, libr3,
# moe-remote, ngram-helper and bc-gfx12 builds need, preinstalled so a run does not apt-get on every start.
#
# Slim on purpose: the builds use only the HIP compiler (rocm-llvm via hipcc), the HIP runtime (amdhip64, found by
# find_library) and the header-only rocWMMA -- no ROCm math libraries -- so this takes the rocm-hip-runtime-dev
# meta-package instead of the full rocm/dev image. The tag is a hash of this file (radiance-build-image.yml).
FROM ubuntu:24.04

ARG ROCM_VERSION=7.2.4

RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl gnupg; \
    mkdir -p /etc/apt/keyrings; \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
        > /etc/apt/sources.list.d/rocm.list; \
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        rocm-hip-runtime-dev rocwmma-dev \
        git cmake ninja-build python3 python3-pip g++-14 gcc-14 ccache; \
    apt-get purge -y --auto-remove curl gnupg; \
    rm -rf /var/lib/apt/lists/*; \
    test -x /opt/rocm/lib/llvm/bin/clang++; \
    test -e /opt/rocm/lib/libamdhip64.so

ENV ROCM_PATH=/opt/rocm
