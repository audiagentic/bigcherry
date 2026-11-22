# Radiance RUM13: native mixed-architecture exact AQL collective test

**Owner:** proposed RUM13 shared Radiance qualification, not LTX.
**Disposition:** self-contained opt-in GPU lab only, not installed, not
a production engine startup/serving path.

This is the **single** supported validation executable after deleting
the C07 standalone direct-HIP launcher. It links the original HIP-enabled
Radiance 2.2 / ABI44 native core as a **whole archive** with the AQL
backend's kernel registration and actual shared-slot code. It uses the
shipped gfx1100 `libr3.so` and gfx1201 `libr4d.so` native plugins,
`rad_dev_set` / `rad_stream_create` / `rad_dev_peer_launch` /
`rad_stream_sync` and the existing one-/two-shot exact BF16 rows.
No custom collective, peer transport, shims or new runtime configuration
is introduced.

The original C07 direct-HIP probe bypassed native AQL fat-binary
registration and crashed on mixed-card launch. Its immutable receipt
is under `docs/evidence/2026-11-22-ltx25-c07-native-mixed-ar/`.
The passing C08 baseline is under
`docs/evidence/2026-11-22-ltx25-c08-aql-mixed-collective/`.
These historical documents do not define supported new launch modes.

## Configure/build (no GPU required)

Use one **matching** source/build/sdk, not mixed or fallback archives:

```sh
cmake -S tools/lab/radiance/native-mixed-ar   -B "$BC_WORK_ROOT/rum13-native-aql-build"   -DRADIANCE_SOURCE_ROOT="$RADIANCE_22_SOURCE"   -DRUM13_NATIVE_CORE_LIB="$RADIANCE_22_HIP_CORE"   -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build "$BC_WORK_ROOT/rum13-native-aql-build" --parallel 2
ctest --test-dir "$BC_WORK_ROOT/rum13-native-aql-build" --output-on-failure
```

The build enforces the pinned **2.2.0** source and HIP-on receipt;
one executable is produced: `radiance_mixed_ar_gate`. The negative
CTest rejects three independent ranks placed on the same GPU *before*
loading the plugin or creating a GPU device/stream. Distinct device
ordinals are required; negative ordinals are also refused.

## Actual GPU check (short lease, run exact native AQL only)

```sh
agent-lease acquire gpu --ttl 15m --est 5m --wait --   python3 tools/lab/radiance/native-mixed-ar/run_mixed_ar.py     --binary "$BC_WORK_ROOT/rum13-native-aql-build/radiance_mixed_ar_gate"     --sdk "$RADIANCE_22_SDK" --output "$BC_WORK_ROOT/rum13-aql-proof"     --devices 0,1,2 --visible 0,1,2     --row ar_oneshot_3rank_exact_host --timeout 40
agent-lease acquire gpu --ttl 15m --est 5m --wait --   python3 tools/lab/radiance/native-mixed-ar/run_mixed_ar.py     --binary "$BC_WORK_ROOT/rum13-native-aql-build/radiance_mixed_ar_gate"     --sdk "$RADIANCE_22_SDK" --output "$BC_WORK_ROOT/rum13-aql-proof"     --devices 0,1,2 --visible 0,1,2     --row ar_twoshot_3rank_exact_host --timeout 40
```

The runner writes JSON and log per selected row, physical GPU ordinals
and ROCm visibility. Native validation covers eleven 1/4/16/64-row
BF16 trials per rank at widths 2048 and 4096, disjoint head spans
8/8/16 and 12/12/8, overlapping partial sums, and in-/out-of-place
modes. The exact ascending-rank FP32 sum is rounded once to BF16.
Require **33/33 exact comparisons per variant** across three *distinct*
physical gfx1100/gfx1100/gfx1201 cards (66/66 total), shared-slot
participation by both libraries, and no process or lease left behind.

Do **not** use `--devices 0,0,0` or `0,0,1` to run this spin-barrier
multi-rank collective: three ranks on fewer physical GPUs are not
this independent-card test's intended topology. The former stalled
and timed out in a 2026-11-22 control; the new binary refuses that
placement. A future true homogeneous control must use distinct
physical GPUs or a kernel-specific single-device test; this lab does
not fabricate a 3-rank intra-device transport.

## Disposition / nonclaims

RUM13 owns this *generic qualification harness*; RUM06 alone owns
production rendezvous, fault/drain and collective lifetime, RUM05
owns native device admission, and LTX02 owns issued checkpoint-backed
`RadCtx` Q/K RMS/self-attention numerical comparison. These are
different gates. Exact AQL collectives here do not prove full LTX
TP3 model inference or production serving safety. After every
GPU test the lease must be released and `rocm-smi --showpids` empty.
