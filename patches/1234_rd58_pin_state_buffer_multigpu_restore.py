"""RD58 (UP-HIP-003): pin the host state buffer during multi-GPU
prompt-cache/checkpoint state restore, to work around a real, still-open
ROCm runtime defect that faults an async H2D copy from pageable host
memory when 2+ devices share one process.

Provenance (group 'rdna-boosts' patches are external backports; the
machine-readable PROVENANCE dict below is cross-checked against
external-sources.toml by tools/tests/test_external_sources.py):

  source:            ggml-org-llama-cpp (upstream PR, not merged)
  repo:              https://github.com/ggml-org/llama.cpp
  locator:           PR #27405 (closed, not merged)
  commit:            source PR's HEAD; see external-sources.toml for the
                     exact tracked SHA
  title:             "cuda : pin the host state buffer during multi-GPU
                     state restore"
  reviewed:          2026-08-23/24 (RD58 real-hardware validation pass)

What it does: llama_context::state_seq_set_data() restores prompt-cache/
checkpoint sequence state via an async H2D copy from a pageable host
buffer (llama_io_read_host). ROCm/rocm-systems#4817 (still open as of
2026-08-22, reported by a ggml HIP-backend maintainer, explicitly names
RX 7900 XTX among affected hardware) is a real runtime defect: with 2+
devices in one process, ROCm's on-the-fly pageable-memory mapping for an
async H2D copy can be torn down mid-transfer, faulting the SDMA engine.
The fault is intermittent and load-independent (identical binary/config
passes N times, faults at N+1) -- indistinguishable from flaky hardware
without this diagnosis. Registering the restore buffer as portable
pinned host memory for the duration of the restore keeps the mapping
stable; the guard is declared before the `io` object so unregistration
only runs after ~llama_io_read_host's deferred copies have flushed.

Upstream-fork measured (PR's own data, 2x MI210): unpinned faults after
2 restores; pinned survives 140 restores / 12 rounds / 0 faults,
temp-0 output unchanged.

Why not merged upstream: closed by a maintainer 2026-08-20 with "Fix it
in ROCm. cc @ggml-org/amd", redirecting to rocm/rocm-systems#4817 --
not because the diagnosis or fix was disputed. The underlying ROCm
defect remains open and unresolved, so the workaround is still the only
available mitigation at the application layer.

Porting notes (RD58's own agreed delta from the source PR, decided
2026-08-24 before implementation):
  - Ported the RAII lifetime mechanism, the host-only path (never
    LLAMA_STATE_SEQ_FLAGS_ON_DEVICE), the >=1 MiB size threshold, the
    proc-address lookup via ggml_backend_reg_get_proc_address() (the
    same pattern already used elsewhere in this exact file, e.g. for
    ggml_backend_set_n_threads/set_abort_callback), portable
    registration, and the guard-before-io destruction ordering --
    unchanged from upstream.
  - DEVIATES from the source PR on failure logging. Upstream emits
    LLAMA_LOG_WARN both on success AND on reg_fn()==false. At this
    pin, ggml_backend_register_host_buffer()'s CUDA/HIP implementation
    itself owns the GGML_CUDA_REGISTER_HOST opt-in policy -- if the env
    var is unset, reg_fn() returns false by design, not because
    anything failed. llama-context.cpp asking a backend "can you
    register this?" and getting "no" is not an error condition; a
    literally-ported warning would fire on every ordinary restore where
    the opt-in knob is simply unset (the default), which is misleading.
    Kept the success diagnostic (useful as a real-hardware activation
    proof that a given run actually exercised the pinned path) and
    silenced the false-path warning entirely. This does not distinguish
    "deliberately disabled" from "opted in but hipHostRegister itself
    failed" -- accepted per this design decision, not deferred: adding
    a tri-state result-with-reason contract to the backend registration
    ABI is out of RD58's scope. The success diagnostic's presence is
    the activation invariant this item's own hardware validation
    relies on: no diagnostic means the treatment path was not
    exercised, and that run cannot count as pinned evidence.
  - Kept the source PR's `break` after the first backend whose reg/
    unreg proc-address pair resolves, even when registration returns
    false -- do not fall through to try a second GPU's portable
    registration for the same buffer, which would make the fail-open
    behavior device-context-dependent (not a semantic change from
    upstream, called out explicitly since it is easy to miss when
    reading only the diff).

Hardware status: real-hardware validation pass in progress on Brutus
(2+ HIP devices) -- see this item's plan-item notes for reproduction/
validation results. STATE stays "untested" pending that evidence.

Isolation and promotion (first-sweep policy, matching existing RD
items): GROUP 'rdna-boosts' + STATE 'untested' keeps this OUT of the
production 'framework' and 'validated-enhancements' patch-sets until
real-hardware validation lands.
"""

import re

from bigcherry.patcher import Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "untested"

PROVENANCE = {
    "source-id": "davetha-llama-cpp",
    "plan-item": "RD58",
    "fork-commit": "58025e01afd44ee06ea6fb23c7ccbdf38e0c34d5",
    "fork-commit-title": "cuda : pin the host state buffer during multi-GPU state restore",
    "snapshot-head": "58025e01afd44ee06ea6fb23c7ccbdf38e0c34d5",
    "snapshot-base": "58025e01afd44ee06ea6fb23c7ccbdf38e0c34d5",
    "adaptations": [
        "PR #27405 was closed without merging (redirected to the still-open "
        "ROCm defect it works around, rocm/rocm-systems#4817), so there is "
        "no landed upstream commit SHA to track -- referenced by PR number. "
        "Removed the unconditional LLAMA_LOG_WARN on reg_fn()==false "
        "(fires on every ordinary restore where GGML_CUDA_REGISTER_HOST is "
        "simply unset, not an error); kept the success diagnostic as the "
        "real-hardware activation proof this item's validation relies on. "
        "No other behavior change from the source PR.",
    ],
}

_ANCHOR_OLD = """size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;"""

_ANCHOR_NEW = """size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    // bigcherry (RD58): pin the host state buffer for the duration of the
    // restore. With 2+ devices in one process, ROCm's on-the-fly mapping of
    // pageable memory for async H2D copies can be torn down mid-transfer and
    // the SDMA engine faults inside the copy's source range (ROCm/rocm-systems
    // #4817, still open). Registering the range as portable pinned memory
    // keeps the mapping stable. Declared before `io` so unregistration runs
    // only after the deferred copies in ~llama_io_read_host have flushed.
    struct rd58_pin_guard_t {
        void * ptr = nullptr;
        void (* unreg)(void *) = nullptr;
        ~rd58_pin_guard_t() { if (ptr && unreg) { unreg(ptr); } }
    } rd58_pin_guard;
    if (!(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && size >= 1u << 20) {
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (!reg) {
                continue;
            }
            auto * reg_fn   = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_register_host_buffer");
            auto * unreg_fn = (void (*)(void *))         ggml_backend_reg_get_proc_address(reg, "ggml_backend_unregister_host_buffer");
            if (reg_fn && unreg_fn) {
                // bigcherry (RD58): deliberately silent on reg_fn()==false --
                // the backend's own GGML_CUDA_REGISTER_HOST opt-in gate
                // returning "no" is not an error, it's the default. Only the
                // success path logs, so its presence is the real-hardware
                // activation proof this item's validation relies on.
                if (reg_fn(const_cast<uint8_t *>(src), size)) {
                    rd58_pin_guard.ptr   = const_cast<uint8_t *>(src);
                    rd58_pin_guard.unreg = unreg_fn;
                    LLAMA_LOG_WARN("%s: pinned state buffer (%zu bytes) for restore\\n", __func__, size);
                }
                break;
            }
        }
    }
    std::unique_ptr<llama_io_read_i> io;"""


PATCH = FilePatch(
    path="src/llama-context.cpp",
    description="Pin the host state buffer during multi-GPU prompt-cache/"
                "checkpoint state restore to work around ROCm/rocm-systems "
                "#4817 (ggml-org/llama.cpp PR #27405 / RD58)",
    edits=(
        Edit(
            id="rd58-pin-state-buffer-multigpu-restore",
            anchor=re.escape(_ANCHOR_OLD),
            rationale="state_seq_set_data(): host-register the restore "
                      "buffer for the duration of the restore, guarded on "
                      "size and the ON_DEVICE flag, to avoid an intermittent "
                      "ROCm async-H2D-copy fault with 2+ devices",
            mode="replace",
            text=_ANCHOR_NEW,
            guard=r"rd58_pin_guard_t",
        ),
    ),
)

PATCHES = [PATCH]
