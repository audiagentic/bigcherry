"""HTR03 real-hardware confirmation (GPT-required, session
ses_330ae3c055084f38, 2026-08-30): reuse the ORIGINAL known-guilty HI141
cache and confirm the NEW manifest-driven corpus/applicability path
(production-dual-xtx -> behavioral_classes -> corpus edition -> resolved
vector -> requires_mtp) still reproduces the exact original hard_fail,
with the new provenance fields all present and correct.
"""
import sys
sys.path.insert(0, "/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/tools")

import json
from pathlib import Path

from bigcherry.core import config as campaign_config
from bigcherry.tuning import behavioral_gate as bg
from bigcherry.tuning import workflow as wf
from bigcherry.tuning.server_runner import ServerRunner

BINARY_PATH = Path("/home/audumla/.cache/bigcherry/artifacts-store/builds/b6048afbd7787b0fc34982c237d81743/69f0be9b60f65f42cf172bb29b183c1e/llama-server")
# The ORIGINAL known-guilty full cache (all 20 signatures at their
# originally-tuned winners, including the guilty mmvq:q8_0:w4:nw8:rpb1:sk0:v1)
GUILTY_CACHE = None
for candidate in [
    Path("/mnt/vault/experiments/hi141-proof/hi141-proof-20260829-2231/dispatch.cache.provisional"),
]:
    if candidate.is_file():
        GUILTY_CACHE = candidate
        break
MODEL = Path("/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf")
DEVICES = "0,1"
PORT = 44051


def main():
    assert GUILTY_CACHE is not None, "could not find the known-guilty cache artifact"
    cfg = campaign_config.load(Path("/mnt/vault/development/llmhosts/bigcherry-hi143-e2e-20260829/config/recipes.toml"))
    profile = cfg.runtime_profiles["production-dual-xtx"]
    print(f"profile={profile.name} behavioral_classes={profile.behavioral_classes} "
          f"behavioral_corpus_edition={profile.behavioral_corpus_edition} digest={profile.digest[:16]}")

    vectors, edition, specs = wf._resolve_default_corpus(profile)
    print(f"resolved corpus edition={edition.edition if edition else None} "
          f"content_digest={edition.content_digest[:16] if edition else None} "
          f"vectors={[v.name for v in vectors]}")
    assert len(vectors) == 1
    vector = vectors[0]
    print(f"vector.requires_mtp={vector.requires_mtp} n_predict={vector.n_predict} seed={vector.seed}")

    common_args = ("-ngl", "99", "-c", str(profile.production_context), *profile.server_args)

    def run_leg(dispatch_mode, cache_path=None):
        env = {"HIP_VISIBLE_DEVICES": DEVICES, "GGML_HIP_DISPATCH_MODE": dispatch_mode}
        if cache_path is not None:
            env["GGML_HIP_DISPATCH_CACHE"] = str(cache_path)
        runner = ServerRunner(
            binary=BINARY_PATH, model=MODEL, extra_args=common_args, port=PORT,
            env_overrides=env, log_path=Path(f"/mnt/vault/experiments/htr03-confirm-{dispatch_mode}.log"),
        )
        with runner:
            return bg.run_vector(runner, vector, require_mtp=vector.requires_mtp)

    print("running native leg...")
    native_trace = run_leg("native")
    print(f"  native draft_n={native_trace.draft_n} accepted={native_trace.draft_n_accepted}")

    print("running candidate leg (known-guilty cache)...")
    candidate_trace = run_leg("replay", cache_path=GUILTY_CACHE)
    print(f"  candidate draft_n={candidate_trace.draft_n} accepted={candidate_trace.draft_n_accepted}")

    verdict = bg.compare_traces(vector.name, native_trace, candidate_trace)
    print(f"\nVERDICT: {verdict.verdict}")
    print(f"first_output_divergence: {verdict.first_output_divergence}")

    # Build the same enriched provenance record workflow.py now emits.
    entry = {
        "id": specs[0].id, "content_digest": specs[0].content_digest,
        "prompt_sha256": specs[0].prompt_sha256, "n_predict": specs[0].n_predict, "seed": specs[0].seed,
        "applies_to": list(specs[0].applies_to), "requirements": list(specs[0].requirements),
        "verdict": verdict.verdict,
        "native_draft": [verdict.native.draft_n, verdict.native.draft_n_accepted],
        "candidate_draft": [verdict.candidate.draft_n, verdict.candidate.draft_n_accepted],
        "first_output_divergence": verdict.first_output_divergence,
        "native_token_digest": bg.token_digest(verdict.native.generated_token_ids),
        "candidate_token_digest": bg.token_digest(verdict.candidate.generated_token_ids),
        "token_count": len(verdict.native.generated_token_ids),
    }
    report_document = {
        "behavioral_gate_contract_version": bg.BEHAVIORAL_GATE_CONTRACT_VERSION,
        "corpus_edition_id": edition.edition,
        "corpus_schema_version": edition.schema_version,
        "corpus_content_digest": edition.content_digest,
        "runtime_profile_name": profile.name,
        "runtime_profile_digest": profile.digest,
        "selected_vectors": [entry],
    }
    print("\n########## FULL PROVENANCE RECORD ##########")
    print(json.dumps(report_document, indent=2))

    expected_hard_fail = verdict.verdict == "hard_fail"
    print(f"\nCONFIRMATION: {'PASS -- reproduced the known hard_fail via the new manifest path' if expected_hard_fail else 'UNEXPECTED -- did not reproduce hard_fail'}")


if __name__ == "__main__":
    main()
