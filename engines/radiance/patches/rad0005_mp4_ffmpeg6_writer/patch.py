"""Let radiance 1.8.0's MP4 writer build against FFmpeg 6.

core/mm/mp4.cpp (new in 1.8.0) hands avio_alloc_context a write callback taking `const uint8_t*`, the FFmpeg 7
(libavformat 61) signature. FFmpeg 6.1 (Ubuntu 24.04, dev) declares it `uint8_t*` and the file does not compile.
The patch picks the parameter type by LIBAVFORMAT_VERSION_MAJOR; the body only reads the buffer, so behaviour is
identical on both.
"""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="core/mm/mp4.cpp",
    description="Match the avio write callback's buffer type to the libavformat in use",
    edits=(
        Edit(
            id="mem-write-signature",
            anchor=r"^int mem_write\(void\* op, const uint8_t\* buf, int n\) \{$",
            mode="replace",
            text="""#if LIBAVFORMAT_VERSION_MAJOR >= 61
int mem_write(void* op, const uint8_t* buf, int n) {
#else
int mem_write(void* op, uint8_t* buf, int n) {
#endif""",
            guard=r"LIBAVFORMAT_VERSION_MAJOR >= 61",
            expect_matches=1,
            rationale="long-anchor: a single line, the one avio write callback in the file; the signature changed in libavformat 61",
        ),
    ),
)
