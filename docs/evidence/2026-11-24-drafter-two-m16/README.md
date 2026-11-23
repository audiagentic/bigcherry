# VTP37 — IQR M17–64 NT1 slicing (gfx1100)

### Stage 1 — full native build and resource census
- Full native libr3 plugin built and linked: `R3_NATIVE=ON`, `R3_ONLY` empty, no excluded units (115 included).
- One named 16-token tile constant shared by the device kernel, launcher and sliced dispatch.
- New route limited to rotated `i8a8h` B4-table BF16-scale M17–64; other forms and upstream gfx1201 unchanged.
- Stock/rewrite IQR code-object metadata: 840 signatures, zero added/removed/changed resource tuples.
- AQL chained-slice accounting backed by per-kernel packet semantics in source; live AQL tape playback not tested.
