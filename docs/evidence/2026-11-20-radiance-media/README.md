# Image and video input in radiance builds, and what each mode costs

Until now the radiance lane built without a media decoder: CMake's `RAD_WITH_FFMPEG` was `AUTO`, FFmpeg lives under
`/opt/ffmpeg6` on the agent container (not on pkg-config's path) and was not installed in the CI build image, so the engine logged
"decoders absent" while still keeping the 27-layer vision tower in VRAM (879 MiB a rank). Every image request was refused.

## Change

- `engines/radiance/recipes.toml`: `RAD_WITH_FFMPEG = "ON"`, so a missing FFmpeg is a configure error, not a silent text-only build.
- `.github/docker/radiance-build/Dockerfile`: pkg-config and the five libav*-dev packages (Ubuntu's full FFmpeg, PNG included).
- `tools/env/bigcherry-env.sh`: puts `/opt/ffmpeg6` on `CMAKE_PREFIX_PATH` when it exists.
- `tools/lab/radiance/image_probe.py` (+ test): draws a red disc and green square, asks for their colours.

Lane build with the change: configure reports `media decoding via libavformat 60.16.101`.

## Modes (TP3 27B, `qwen27b-tp3-int8-556-long`, 3 R9700-class cards, same config, fresh start each)

| mode | how | image request | VRAM/card (0,1,2 MiB) | KV tokens per rank (own split) |
|---|---|---|---|---|
| on, tower in VRAM | default | answered ("red ... green") | 21352, 21352, 32439 | 1,253,943 |
| on, tower in RAM | `--set vision.vision_tower_placement=ram` | answered | 20910, 20910, 32437 | 1,310,199 (+4.5%) |
| off | `--set vision.mm_max_patches=0` | HTTP 501 "requires a vision encoder; this model was loaded without one" | 20838, 20838, 32481 | 1,319,543 (+5.2%) |

Card 2 is full in every mode (it holds the dearest share and the cache takes what is left); the freed memory shows on cards 0 and 1 as
more KV. RAM placement costs a host pinned pool (about 0.9 GiB) and returned the answer in 1.2 s on a first request, 0.3 s after
the tower was warm in VRAM mode; no throughput comparison was made. Use `--set` (no layout-name change, so live-policy keys stay).

## Known gap: PNG

The agent container's `/opt/ffmpeg6` (image stage `ffmpeg6` in the container Dockerfile, a plain `./configure` without zlib-dev)
has no `png` decoder, so a PNG is refused with "the media holds no video stream". JPEG, BMP, WebP, GIF, AV1, H.264, HEVC and VP9
are present. CI's Ubuntu FFmpeg has PNG. Fix for the container: add `zlib1g-dev` to that stage's apt list (or `--enable-zlib`
with zlib present) and rebuild. The probe sends BMP by default (`--format jpeg|png` available).
