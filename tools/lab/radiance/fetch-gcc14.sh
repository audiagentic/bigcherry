#!/bin/bash
# Radiance needs g++ 14 as its host compiler. Where the host only has g++ 13 and no root access is wanted, this
# unpacks Ubuntu's g++-14 packages into a private directory: nothing is installed and the system compiler is unchanged.
# The packages' shared-library links point at relative system paths that do not exist under the private root, so they
# are re-pointed at the system runtime libraries (Ubuntu 24.04's libstdc++6 / libgomp1 are already the gcc-14 builds);
# without that the linker falls back to the static libgomp.a and shared objects fail to link.
# Usage: fetch-gcc14.sh [dest]      default dest: /mnt/data/bigcherry-work/toolchains/gcc14
set -eu
T=${1:-/mnt/data/bigcherry-work/toolchains/gcc14}
mkdir -p "$T/debs"
cd "$T/debs"
apt-get download g++-14 gcc-14 cpp-14 g++-14-x86-64-linux-gnu gcc-14-x86-64-linux-gnu cpp-14-x86-64-linux-gnu \
    libstdc++-14-dev libgcc-14-dev gcc-14-base
for d in *.deb; do dpkg -x "$d" "$T/root"; done
D=$T/root/usr/lib/gcc/x86_64-linux-gnu/14
for l in $(find "$D" -maxdepth 1 -type l); do
    [ -e "$l" ] && continue
    n=$(basename "$(readlink "$l")")
    if [ -e "/usr/lib/x86_64-linux-gnu/$n" ]; then
        ln -sfn "/usr/lib/x86_64-linux-gnu/$n" "$l"
    else
        echo "no system library for $(basename "$l") ($n)" >&2
    fi
done
"$T/root/usr/bin/x86_64-linux-gnu-g++-14" --version | head -1
echo "GCC14_BIN=$T/root/usr/bin"
