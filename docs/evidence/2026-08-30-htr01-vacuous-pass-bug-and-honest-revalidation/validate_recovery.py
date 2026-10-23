"""HTR01 real-hardware validation: run the actual recovery search against
the EXACT real hard_fail captured by hi141-proof-20260829-2231 (already on
disk -- no new tune/record/correctness stages needed, just recovery
against already-collected data), confirming it either ships a real reduced
cache or correctly falls back to native for the minimum necessary
signatures, with a mandatory full-corpus validation gating either outcome.
"""
import sys
sys.path.insert(0, "/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/tools")

import json
from pathlib import Path

from bigcherry.tuning import behavioral_gate as bg
from bigcherry.tuning import recovery as rec
from bigcherry.tuning import workflow as wf

CAMPAIGN_DIR = Path("/mnt/vault/experiments/hi141-proof/hi141-proof-20260829-2231")
PROMOTED_PATH = CAMPAIGN_DIR / "promoted.jsonl"
BINARY_PATH = Path("/home/audumla/.cache/bigcherry/artifacts-store/builds/b6048afbd7787b0fc34982c237d81743/69f0be9b60f65f42cf172bb29b183c1e/llama-server")
MANIFEST_PATH = Path("/home/audumla/.cache/bigcherry/artifacts-store/runs/hi141-proof-20260829-2231-replay-b439ea406a0b/generate/hip-autotune-manifest.json")
SOURCE_ROOT = Path("/home/audumla/.cache/bigcherry/sources/38c99c0a48996a2e9c1c717a3d492060")
GGML_H = SOURCE_ROOT / "ggml" / "include" / "ggml.h"
CORRECTNESS_BINARY_PATH = Path("/home/audumla/.cache/bigcherry/artifacts-store/builds/f3e14c088e667bfc1c9d533422150c5d/70b7ef2748113d6e9901115952e1f822/test-backend-ops")
CORRECTNESS_VENDOR_ROOT = Path("/home/audumla/.cache/bigcherry/sources/7ed939cded44ad21629d167eeb4613a0")
MODEL = Path("/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf")
DEVICES = "0,1"
WORKDIR = Path("/mnt/vault/experiments/hi141-recovery-validation")
WORKDIR.mkdir(parents=True, exist_ok=True)

COMMON_ARGS = (
    "-ngl", "99", "-c", "64000",
    "-sm", "tensor", "--flash-attn", "on",
    "--ubatch-size", "512", "--batch-size", "2048",
    "--threads", "8", "--parallel", "1",
    "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
    "-ctkd", "q8_0", "-ctvd", "q8_0",
)


def main():
    assert BINARY_PATH.is_file(), f"missing binary: {BINARY_PATH}"
    assert MANIFEST_PATH.is_file(), f"missing manifest: {MANIFEST_PATH}"
    assert GGML_H.is_file(), f"missing ggml.h: {GGML_H}"
    assert PROMOTED_PATH.is_file(), f"missing promoted.jsonl: {PROMOTED_PATH}"

    assignments = wf._load_signature_assignments(PROMOTED_PATH)
    print(f"loaded {len(assignments)} promoted non-native signature assignments")
    for dispatch, a in list(assignments.items())[:5]:
        print(f"  {dispatch[:12]}... current={a.current_candidate} alternatives={len(a.alternatives)}")

    corpus = [bg.load_hi141_regression_vector()]

    dispatch_db = CAMPAIGN_DIR / "tune.sqlite"
    assert dispatch_db.is_file(), f"missing dispatch_db: {dispatch_db}"
    executor = rec.AssignmentExecutor(
        binary_path=BINARY_PATH, model_path=MODEL, devices=DEVICES,
        common_args=COMMON_ARGS, measurements_path=PROMOTED_PATH,
        manifest_path=MANIFEST_PATH, ggml_h_path=GGML_H, workdir=WORKDIR,
        dispatch_db=dispatch_db,
        correctness_binary_path=CORRECTNESS_BINARY_PATH,
        vendor_root=CORRECTNESS_VENDOR_ROOT,
        campaign_run_id="hi141-proof-20260829-2231",
        recovery_run_id="hi141-proof-20260829-2231-recovery-retry2",
    )

    print("\ncapturing native traces + initial candidate-cache gate result (reproducing the original hard_fail)...")
    executor.capture_native_traces(corpus)
    all_winners_overrides = {}  # empty overrides = use campaign's own original winners as-is
    initial_observation = executor.evaluate(
        rec.AssignmentProposal(label="original-full-cache", overrides=all_winners_overrides),
        full_corpus=corpus,
    )
    print(f"initial verdict (should reproduce hard_fail): {initial_observation.verdict}")
    print(json.dumps(initial_observation.report.summary(), indent=2))

    if initial_observation.verdict == "pass":
        print("\nNOTE: this did not reproduce hard_fail (masking or environment drift) -- "
              "recovery has nothing to recover from. Exiting without running recovery.")
        return

    initial_report = initial_observation.report
    strategy = rec.BoundedPairedBisectionStrategy()
    dispatch_hits = frozenset(assignments)

    print(f"\nrunning bounded recovery search (budget={rec.DEFAULT_MAX_RECOVERY_EVALUATIONS})...")
    result = rec.run_recovery(
        executor=executor, strategy=strategy, initial_assignments=assignments,
        initial_report=initial_report, full_corpus=corpus, dispatch_hits=dispatch_hits,
    )

    print("\n########## RECOVERY RESULT ##########")
    print(f"published: {result.published}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"evaluations_used: {result.evaluations_used}")
    print(f"cache_path: {result.cache_path}")
    non_native_final = {k: v for k, v in result.final_overrides.items() if v != "native"}
    native_final = {k: v for k, v in result.final_overrides.items() if v == "native"}
    print(f"final signatures reassigned to an alternative candidate: {len(non_native_final)}")
    for k, v in non_native_final.items():
        print(f"    {k[:12]}... -> {v}")
    print(f"final signatures forced to native: {len(native_final)}")
    for k in native_final:
        print(f"    {k[:12]}... -> native")
    print(f"retune_recommendations: {len(result.retune_recommendations)}")
    for r in result.retune_recommendations:
        print(f"    {r.signature_dispatch[:12]}... reason={r.reason} exhausted={len(r.exhausted_candidates)}")

    total_signatures = len(assignments)
    recovered_count = total_signatures - len(result.final_overrides) + len(non_native_final)
    print(f"\nSUMMARY: of {total_signatures} originally-promoted non-native signatures, "
          f"{total_signatures - len(result.final_overrides)} were never touched (already clean), "
          f"{len(non_native_final)} were reassigned to a real alternative candidate, "
          f"{len(native_final)} fell back to native.")


if __name__ == "__main__":
    main()
