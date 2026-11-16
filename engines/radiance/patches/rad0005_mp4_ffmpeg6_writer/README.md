# rad0005_mp4_ffmpeg6_writer

## Summary

Selects the FFmpeg 6 vs 7 avio_alloc_context write-callback signature so rad_core compiles on Ubuntu 24.04's FFmpeg 6.1.

## Outcome

Validated on the promotion evidence below.

## Detail

### Promotion record

Model: none (build enabler for the MP4 writer, no model path); built against FFmpeg 6.1 (libavformat 60.16.101, /opt/ffmpeg6) on the lab host
Mechanics: tools/tests/patch/test_radiance_rad0005.py (applies on the v1.8.0 fixture) passes and patch-lint is clean on the promotion PR; the patch applies cleanly to v1.17.0 (patch-rebase-check --source radiance: 26 clean).
Activation: compiled core/mm/mp4.cpp with RAD_HAVE_FFMPEG=1 and -I/opt/ffmpeg6/include (the production build's own compile command, g++-14, 2026-11-16): the pristine v1.17.0 file fails with "invalid conversion from int (*)(void*, const uint8_t*, int) to int (*)(void*, uint8_t*, int)" at the avio_alloc_context call; the patched file compiles clean, the `LIBAVFORMAT_VERSION_MAJOR >= 61` guard selecting the FFmpeg 6 signature (libavformat is 60 here).
Identity: the write callback only reads its buffer, so the bytes written are identical on both signatures; the patch changes the parameter's constness and nothing else (diff of three lines). No runtime MP4 write was run: the current production build does not define RAD_HAVE_FFMPEG (RAD_WITH_FFMPEG=AUTO found no FFmpeg flags), so mp4.cpp compiles as its stub there.
A/B: none; a compile-compatibility enabler with no runtime path of its own, so there is no performance or output claim.
