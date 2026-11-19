# LTX-2.5 IQ8R container: local read-only preflight

**Verified 2026-11-20 on the development host.** This is bounded structural evidence, not a decoded-IQ8R validation, a model digest, a kernel test, or a successful video inference.

## Model acquisition is no longer blocking

A regular container is available at the local model root under `ltx-2.5/ltx-2.5-iq8r.rad`, with its matching `ltx-2.5-iq8r.recipe`, README, LICENSE and NOTICE. Refer to it via `${BC_MODEL_ROOT}/ltx-2.5/ltx-2.5-iq8r.rad` or supply that path in the development-only `BC_LTX_IQ8R_MODEL` environment variable. **Never track a host-specific absolute model path, a model binary, or license-gated weight data in BigCherry's Git repository.**

The file's observed regular-file length was **39,578,374,144 bytes**. Its first 248 bytes were inspected using the frozen candidate's `abi/rad_format.h` layout. The header declared exactly the same file length, `RAD1` magic (the file-family marker) and **format version 2**. The string table entry for architecture and model both decode to `ltx2`; `created_by` begins `rad-convert Nov  9 2026 from hfq-ltx/ltx-2.5-full`.

| Header section | Offset (bytes) | Length or count |
| --- | ---: | ---: |
| Strings | 248 | 2,791,151 bytes |
| Metadata | 2,791,400 | 423 entries |
| Weight directory | 2,801,552 | 6,114 entries |
| Weight planes | 3,633,056 | 8,726 entries |
| Encodings | 3,772,672 | 3 entries |
| Vocabulary | 3,774,376 | 6,478,864 bytes |
| Data | 10,485,760 | 39,567,888,384 bytes |

These header values were obtained with bounded reads and file stat only; the 37-GiB data section was **not** read or hashed. They do not prove that any quantized weight plane has the correct bytes. The accompanying converter recipe declares `codes=i8 group=256 scale=u8 scale2=bf16 transform=fwht128` for Gemma, connector and DiT linears, with BF16 exceptions and F32 vocoders. Recipe presence likewise does not prove each actual encoding plane is correct.

## Development-only, model-aware structural check

`tools/tests/lab/test_ltx_tp3_reference.py` contains a fixed-size `RadFileHeader` parser. In ordinary CI it tests a **synthetic sparse fixture** and skips the actual model check. On a host with model access, opt into the real-file check explicitly:

```sh
BC_LTX_IQ8R_MODEL="${BC_MODEL_ROOT}/ltx-2.5/ltx-2.5-iq8r.rad" \
  pytest -q tools/tests/lab/test_ltx_tp3_reference.py
```

This reads only the header and the two short strings. It checks magic, version, byte count, bounds, `ltx2` identity and existence of a nonempty weight directory/plane/encoding table. It does not parse all entries, measure activation memory, allocate GPU memory, execute HIP kernels or provide numerical parity.

When a sandbox permits the approved native binary, run the frozen-source `rad-info --meta`, `rad-info --recipe`, and targeted `rad-info --planes 'dit.blk.*.qkv.weight'` on the same file to inspect declared metadata and **actual** plane layout. `source/tools/rad_info.cpp` explicitly opens containers read-only, without loading plugins. At authoring time the remote shell's command allowlist denied `rad-info`; do not bypass that policy through a subprocess. These native inspections remain **not run**.

## Next evidence unlocked by local model availability

1. Compare exact header and model metadata from `rad-info --meta` with the frozen LTX builder `arch/ltx2/ltx2.cpp`, including video/audio geometries, text layers, quantized linears and bias/gain dtypes.
2. Use `rad-info --planes` or the existing format reader to enumerate QKV and row-parallel output **codes, scale and scale2 planes**, then test actual row-source slicing and per-256-K alignment against `LinearFP8::span_parts`. Keep the reader read-only and CPU-only.
3. Compute and retain a full model SHA-256 only under an agreed 37-GiB disk-read budget; it has not been computed here. The recipe text should also have its own digest for run identity.
4. Request a hardware lease separately before any model load/VRAM or HIP inference test. The one-rank LTX guard must not be relaxed just because the model exists.

The model's README and LICENSE impose the LTX-2.x Community License Agreement. The files' presence is not a change to licensing terms or permission to redistribute them.
