// bigcherry: canonical signature and hardware key construction (HI05).

#include "hip-autotune-signature.h"

// Standards 12.2: gated on GGML_USE_HIP *and* the feature flag. These files
// are picked up by the ggml-cuda/*.cu glob, so without the second condition
// they would compile into every HIP build, including ones that carry none of
// the machinery they depend on.
#if defined(GGML_USE_HIP) && defined(GGML_HIP_DISPATCH)

#include "hip-autotune-blake2b.h"
#include "hip-autotune-build-hash.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

// ------------------------------------------------------------ canonical JSON
//
// Hand-rolled rather than pulled from a library, for one reason: the key order
// must be fixed and obvious. These writers emit keys in alphabetical order by
// construction, matching Python's
// ``json.dumps(..., sort_keys=True, separators=(",", ":"))``. Anyone adding a
// field has to place it in sorted position, which is a visible, reviewable act
// rather than an invisible dependence on struct layout.

namespace {

void append_int(std::string & out, int64_t value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%" PRId64, value);
    out += buffer;
}

void append_key(std::string & out, const char * key, bool first) {
    if (!first) {
        out += ',';
    }
    out += '"';
    out += key;
    out += "\":";
}

void append_int_field(std::string & out, const char * key, int64_t value,
                      bool first = false) {
    append_key(out, key, first);
    append_int(out, value);
}

void append_int_array(std::string & out, const char * key,
                      const int64_t * values, int count) {
    append_key(out, key, false);
    out += '[';
    for (int i = 0; i < count; ++i) {
        if (i != 0) {
            out += ',';
        }
        append_int(out, values[i]);
    }
    out += ']';
}

void append_string_field(std::string & out, const char * key,
                         const char * value, bool first = false) {
    append_key(out, key, first);
    out += '"';
    // The only strings we serialise are hex digests and short ASCII
    // identifiers, so there is nothing here that needs escaping. Anything else
    // would be a bug, and asserting that here beats emitting invalid JSON.
    for (const char * p = value; *p != '\0'; ++p) {
        const char c = *p;
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                       || (c >= '0' && c <= '9') || c == '-' || c == '_'
                       || c == '.' || c == ':';
        out += safe ? c : '_';
    }
    out += '"';
}

} // namespace

std::string ggml_hip_signature_json(const ggml_hip_dispatch_signature_v1 & sig,
                                    bool include_refinements) {
    const bool refined = include_refinements && sig.has_refinements != 0;

    std::string out;
    out.reserve(512);
    out += '{';

    // Alphabetical order. Refinements sort in among the hard fields rather than
    // being appended, so a refined key is not merely "the base key plus a
    // suffix" -- it is a different canonical document, which is what makes the
    // two digests independent.
    if (refined) {
        append_int_field(out, "alignment_class", sig.alignment_class, true);
        append_int_field(out, "dst_type", sig.dst_type);
    } else {
        append_int_field(out, "dst_type", sig.dst_type, true);
    }
    append_int_field(out, "flags", sig.flags);
    append_int_field(out, "fusion", sig.fusion);
    append_int_field(out, "glu_op", sig.glu_op);
    append_int_field(out, "n_expert", sig.n_expert);
    append_int_field(out, "n_expert_used", sig.n_expert_used);
    append_int_array(out, "nb0", sig.nb0, 4);
    append_int_array(out, "nb1", sig.nb1, 4);
    append_int_array(out, "nbd", sig.nbd, 4);
    append_int_array(out, "ne0", sig.ne0, 4);
    append_int_array(out, "ne1", sig.ne1, 4);
    append_int_array(out, "ned", sig.ned, 4);
    if (refined) {
        append_int_field(out, "occupancy_bucket", sig.occupancy_bucket);
        append_int_field(out, "offset_modulo", sig.offset_modulo);
    }
    append_int_field(out, "op", sig.op);
    append_int_field(out, "prec", sig.prec);
    append_int_field(out, "schema_version", sig.schema_version);
    append_int_field(out, "src0_type", sig.src0_type);
    append_int_field(out, "src1_type", sig.src1_type);

    out += '}';
    return out;
}

std::string ggml_hip_hardware_json(const ggml_hip_hardware_key_v1 & hw) {
    std::string out;
    out.reserve(160);
    out += '{';
    append_int_field(out, "architecture_code", hw.architecture_code, true);
    append_int_field(out, "compute_units", hw.compute_units);
    append_int_field(out, "feature_flags", hw.feature_flags);
    append_int_field(out, "schema_version", hw.schema_version);
    append_int_field(out, "shared_memory_per_block", hw.shared_memory_per_block);
    append_int_field(out, "wave_size", hw.wave_size);
    out += '}';
    return out;
}

// ---------------------------------------------------------------- digests

static ggml_hip_digest ggml_hip_digest_of(const std::string & canonical,
                                          const char * person) {
    ggml_hip_digest digest;
    ggml_hip_blake2b(digest.bytes, GGML_HIP_DIGEST_BYTES,
                     canonical.data(), canonical.size(), person);
    return digest;
}

ggml_hip_digest ggml_hip_signature_digest(
        const ggml_hip_dispatch_signature_v1 & sig) {
    return ggml_hip_digest_of(ggml_hip_signature_json(sig, true),
                              GGML_HIP_PERSON_SIGNATURE);
}

ggml_hip_digest ggml_hip_signature_base_digest(
        const ggml_hip_dispatch_signature_v1 & sig) {
    return ggml_hip_digest_of(ggml_hip_signature_json(sig, false),
                              GGML_HIP_PERSON_SIGNATURE);
}

ggml_hip_digest ggml_hip_hardware_digest(const ggml_hip_hardware_key_v1 & hw) {
    return ggml_hip_digest_of(ggml_hip_hardware_json(hw),
                              GGML_HIP_PERSON_HARDWARE);
}

ggml_hip_digest ggml_hip_dispatch_digest(const ggml_hip_digest & hardware,
                                         const ggml_hip_digest & signature,
                                         const char * objective) {
    // Hardware, shape and objective only -- deliberately *not* the manifest
    // hash or the source revision.
    //
    // Standards 13.1 originally put both in here, so that a build against a
    // different candidate set could not resolve to winners measured by the
    // previous one. The intent is right and the mechanism was too blunt: it
    // made every rebuild that touched the catalog or bumped upstream silently
    // discard all tuning, because every key moved. A `replay-slim` build is
    // the extreme case -- slimming necessarily changes the candidate set, so
    // not one winner remained reachable while the cache still reported itself
    // loaded (RV12).
    //
    // What is lost is nothing that was actually protecting correctness. Two
    // checks already do that work, and both act on the entry rather than the
    // key: the loader drops any entry naming a candidate this binary does not
    // carry, and the resolver re-runs `can_execute` before launching a stored
    // winner. A winner from an older catalog is therefore still *valid*; it
    // may merely no longer be *optimal*, having been chosen from a different
    // set of options. That is a staleness signal, and the loader now reports
    // it as one instead of manifesting as a total cache miss with no
    // explanation.
    //
    // Interim measure. HI23 carries the fuller design -- provenance per entry,
    // several generations retained, newest applicable winner preferred -- and
    // that is where the manifest and revision belong.
    std::string out;
    out.reserve(256);
    out += '{';
    append_string_field(out, "hardware",
                        ggml_hip_digest_hex(hardware).c_str(), true);
    append_string_field(out, "objective", objective ? objective : "latency");
    append_string_field(out, "signature",
                        ggml_hip_digest_hex(signature).c_str());
    out += '}';
    return ggml_hip_digest_of(out, GGML_HIP_PERSON_DISPATCH);
}

std::string ggml_hip_digest_hex(const ggml_hip_digest & digest) {
    static const char * hex = "0123456789abcdef";
    std::string out;
    out.resize(GGML_HIP_DIGEST_BYTES * 2);
    for (int i = 0; i < GGML_HIP_DIGEST_BYTES; ++i) {
        out[2 * i + 0] = hex[(digest.bytes[i] >> 4) & 0xf];
        out[2 * i + 1] = hex[(digest.bytes[i] >> 0) & 0xf];
    }
    return out;
}

bool ggml_hip_digest_equal(const ggml_hip_digest & a,
                           const ggml_hip_digest & b) {
    return memcmp(a.bytes, b.bytes, GGML_HIP_DIGEST_BYTES) == 0;
}

// ------------------------------------------------------------ construction

static uint8_t ggml_hip_fusion_kind(const ggml_cuda_mm_fusion_args_host * fusion) {
    if (fusion == nullptr) {
        return GGML_HIP_FUSION_NONE;
    }
    const bool has_gate = fusion->gate != nullptr;
    const bool has_bias = fusion->x_bias != nullptr || fusion->gate_bias != nullptr;
    if (has_gate && has_bias) {
        return GGML_HIP_FUSION_GATE_BIAS;
    }
    if (has_gate) {
        return GGML_HIP_FUSION_GATE;
    }
    if (has_bias) {
        return GGML_HIP_FUSION_BIAS;
    }
    return GGML_HIP_FUSION_NONE;
}

// Strides are recorded in elements rather than bytes. Two tensors of the same
// type with the same element strides are laid out identically, and expressing
// it that way keeps the signature stable if a type's size is ever redefined.
static void ggml_hip_fill_strides(int64_t (&out)[4], const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        for (int i = 0; i < 4; ++i) {
            out[i] = 0;
        }
        return;
    }
    const size_t element = ggml_type_size(tensor->type);
    const size_t block   = (size_t) ggml_blck_size(tensor->type);
    for (int i = 0; i < 4; ++i) {
        // nb[0] is the element (or block) stride; dividing it by the type size
        // would be lossy for quantised types, so it is kept as a byte count and
        // the higher strides are normalised by the block size.
        out[i] = (element == 0)
            ? (int64_t) tensor->nb[i]
            : (int64_t) (tensor->nb[i] * block / element);
    }
}

static void ggml_hip_fill_extents(int64_t (&out)[4], const ggml_tensor * tensor) {
    for (int i = 0; i < 4; ++i) {
        out[i] = tensor != nullptr ? tensor->ne[i] : 0;
    }
}

ggml_hip_dispatch_signature_v1 ggml_hip_make_signature(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        const ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    ggml_hip_dispatch_signature_v1 sig;
    memset(&sig, 0, sizeof(sig));

    sig.schema_version = GGML_HIP_SIGNATURE_SCHEMA_VERSION;
    sig.op             = (uint16_t) dst->op;
    sig.src0_type      = (uint8_t) src0->type;
    sig.src1_type      = (uint8_t) src1->type;
    sig.dst_type       = (uint8_t) dst->type;
    sig.prec           = (uint8_t) ggml_get_op_params_i32(dst, 0);
    sig.fusion         = ggml_hip_fusion_kind(fusion);
    sig.glu_op         = (fusion != nullptr && fusion->gate != nullptr)
                       ? (uint8_t) fusion->glu_op : 0;

    ggml_hip_fill_extents(sig.ne0, src0);
    ggml_hip_fill_extents(sig.ne1, src1);
    ggml_hip_fill_extents(sig.ned, dst);
    ggml_hip_fill_strides(sig.nb0, src0);
    ggml_hip_fill_strides(sig.nb1, src1);
    ggml_hip_fill_strides(sig.nbd, dst);

    uint16_t flags = 0;
    if (ggml_is_contiguous(src0)) flags |= GGML_HIP_SIG_SRC0_CONTIGUOUS;
    if (ggml_is_contiguous(src1)) flags |= GGML_HIP_SIG_SRC1_CONTIGUOUS;
    if (ggml_is_contiguous(dst))  flags |= GGML_HIP_SIG_DST_CONTIGUOUS;

    if (ids != nullptr) {
        flags |= GGML_HIP_SIG_HAS_IDS;
        // Standards 11.2: MUL_MAT_ID carries its own semantic fields. Expert
        // counts change which candidates are even legal, so they are hard
        // identity, not a refinement.
        sig.n_expert      = src0->ne[2];
        sig.n_expert_used = ids->ne[0];
    }

    // HI118: ggml_hip_fusion_kind()'s coarse fusion/glu_op fields collapse
    // x_bias and gate_bias into one "has_bias" bit -- record which specific
    // fusion tensor(s) are present so a consumer reconstructing the fused
    // computation (HI119) knows exactly what to synthesize. Geometry itself
    // needs no new fields here: mmvq.cu's own GGML_ASSERT calls prove gate
    // always shares src0's type/stride, and both biases are always F32 sized
    // by dst.ne[0]/n_expert (fields this signature already records).
    if (fusion != nullptr) {
        if (fusion->x_bias      != nullptr) flags |= GGML_HIP_SIG_FUSION_X_BIAS;
        if (fusion->gate_bias   != nullptr) flags |= GGML_HIP_SIG_FUSION_GATE_BIAS;
        if (fusion->x_scale     != nullptr) flags |= GGML_HIP_SIG_FUSION_X_SCALE;
        if (fusion->gate_scale  != nullptr) flags |= GGML_HIP_SIG_FUSION_GATE_SCALE;
        // the struct's sixth fusion field (dst_gate) is deliberately NOT
        // read here -- see hip-autotune-types.h's GGML_HIP_SIG_FUSION_X_
        // SCALE/GATE_SCALE comment: it only exists on this struct under the
        // experimental, non-default patch RD12 (patches/1205_rd12_paired_
        // mmvq_dual_output), and a build without that patch (confirmed on
        // real Brutus hardware) fails to compile against it at all.
    }

    if (src0->ne[2] != 0 && dst->ne[2] != src0->ne[2]) {
        flags |= GGML_HIP_SIG_BROADCAST_CH;
    }
    if (src0->ne[3] != 0 && dst->ne[3] != src0->ne[3]) {
        flags |= GGML_HIP_SIG_BROADCAST_SMP;
    }

    // The condition upstream uses to force the BLAS path. It changes which
    // families are reachable at all, so it belongs in the signature -- two
    // otherwise identical operations with different answers here are not
    // interchangeable.
    if (src0->buffer != nullptr
            && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0)
            && src0->view_src != nullptr) {
        flags |= GGML_HIP_SIG_BAD_PADDING;
    }

    sig.flags = flags;

    // Refinements stay off until a measurement proves one changes the winner
    // (standards 5.5). Promoting a refinement is a deliberate act, because
    // every promotion splits one measured signature into several unmeasured
    // ones.
    sig.has_refinements = 0;

    return sig;
}

// ------------------------------------------------------------- hardware key

int ggml_hip_architecture_code_from_cc(int cc) {
    if (!GGML_CUDA_CC_IS_AMD(cc)) {
        // NVIDIA and Moore Threads parts have no bit in the AMD architecture
        // mask, so nothing matches and dispatch falls back to native.
        return GGML_HIP_ARCH_UNKNOWN;
    }
    // For AMD targets the low bits of `cc` are the gfx identifier. The mapping
    // table is generated alongside the enum, so adding a part is a one-line
    // change in autotune_schema.py.
    return ggml_hip_arch_code_from_gfx(cc - GGML_CUDA_CC_OFFSET_AMD);
}

ggml_hip_hardware_key_v1 ggml_hip_make_hardware_key(int device) {
    const auto & info = ggml_cuda_info().devices[device];

    ggml_hip_hardware_key_v1 hw;
    memset(&hw, 0, sizeof(hw));

    hw.schema_version = GGML_HIP_HARDWARE_SCHEMA_VERSION;
    hw.architecture_code = (uint16_t) ggml_hip_architecture_code_from_cc(info.cc);
    hw.wave_size = (uint16_t) info.warp_size;
    hw.compute_units = (uint16_t) info.nsm;
    hw.shared_memory_per_block = (uint32_t) info.smpbo;

    uint32_t features = 0;
    if (amd_wmma_available(info.cc))          features |= GGML_HIP_FEATURE_WMMA;
    if (amd_mfma_available(info.cc))          features |= GGML_HIP_FEATURE_MFMA;
    if (fp16_mma_hardware_available(info.cc)) features |= GGML_HIP_FEATURE_FP16_MMA;
    if (ggml_cuda_highest_compiled_arch(info.cc) >= GGML_CUDA_CC_DP4A) {
        features |= GGML_HIP_FEATURE_DP4A;
    }
#ifdef GGML_HIP_GRAPHS
    features |= GGML_HIP_FEATURE_GRAPHS;
#endif
    hw.feature_flags = features;

    return hw;
}

#endif // GGML_USE_HIP && GGML_HIP_DISPATCH
