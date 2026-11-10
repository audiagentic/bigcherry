"""Offline VTP06 BF16-vs-IQ6R KLD admission.

Consumes the *native Radiance* --kld-record/kld.json and --kld-out
report.json + report.json.rows artifacts. Does not run a model, invent
quantization results, or promote a production checkpoint. Thresholds
are an explicit, versioned engineering proposal, not measured fact.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "VTP06_KLD_GATE_V1"
POLICY_SCHEMA = "VTP06_KLD_POLICY_PROPOSAL_V1"
MIB = 1024 * 1024


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, label: str, *, lo: float = 0.0, hi: float = 1e12) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label}: numeric field expected")
    val = float(value)
    if not math.isfinite(val) or val < lo or val > hi:
        raise ValueError(f"{label}: out of bounds/nonfinite")
    return val


def load_policy(path: Path) -> dict:
    p = json.loads(path.read_text())
    if p.get("schema") != POLICY_SCHEMA or p.get("authority") != "PROPOSED_NOT_APPROVED":
        raise ValueError("only explicit unapproved policy proposals supported")
    fields = {
        "min_positions": (1, 1e9),
        "min_source_positions": (1, 1e9),
        "min_top1": (0, 1),
        "min_source_top1": (0, 1),
        "max_mean_kld": (0, 10),
        "max_p99_kld": (0, 100),
        "max_p999_kld": (0, 100),
        "max_source_mean_kld": (0, 10),
        "max_ppl_ratio": (1, 10),
    }
    for key, (lo, hi) in fields.items():
        _number(p.get(key), key, lo=lo, hi=hi)
    return p


def verify_corpus(bundle_dir: Path, checkpoint: Path) -> tuple[dict, int, set[str]]:
    manifest = json.loads((bundle_dir / "bundle.json").read_text())
    if (manifest.get("schema") != "VTP06_CORPUS_V1"
            or manifest.get("status") != "STRUCTURAL_CANDIDATE_NOT_NUMERICALLY_QUALIFIED"
            or manifest.get("overview", {}).get("mode") != "candidate"
            or Path(manifest.get("checkpoint", "")).resolve() != checkpoint.resolve()):
        raise ValueError("quality report requires a candidate, not pilot, corpus bundle")
    if sha_file(checkpoint / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("BF16 tokenizer mismatch")
    if sha_file(checkpoint / "tokenizer_config.json") != manifest["tokenizer_config_sha256"]:
        raise ValueError("BF16 tokenizer configuration mismatch")
    holdout = bundle_dir / "holdout.jsonl"
    if sha_file(holdout) != manifest["holdout_sha256"]:
        raise ValueError("evaluation holdout changed since corpus admission")
    if sha_file(bundle_dir / "calibration.jsonl") != manifest["calibration_sha256"]:
        raise ValueError("calibration corpus changed")
    sources: Counter[str] = Counter()
    with holdout.open(encoding="utf-8") as stream:
        for raw in stream:
            row = json.loads(raw)
            sources[row["source"]] += 1
    docs = manifest["overview"]["documents"]["holdout"]
    if sum(sources.values()) != docs or docs < 50 or not sources:
        raise ValueError("holdout document census mismatched or too small")
    return manifest, docs, set(sources)


def validate_rows(path: Path, positions: int, stats: dict) -> dict:
    """Stream all rows [KL, candidate NLL, BF16 NLL, top1] as LE F32."""
    if path.stat().st_size != positions * 16:
        raise ValueError("Radiance .rows byte count mismatches reference positions")
    count = 0
    total_kl = total_q = total_p = 0.0
    agree = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 * MIB), b""):
            if len(block) % 16:
                raise ValueError("unaligned KLD row block")
            values = array.array("f")
            values.frombytes(block)
            if sys.byteorder != "little":
                values.byteswap()
            for i in range(0, len(values), 4):
                kl, nq, np, t1 = values[i:i + 4]
                if any(not math.isfinite(v) for v in (kl, nq, np, t1)):
                    raise ValueError("nonfinite KLD row")
                if kl < -1e-3 or t1 not in (0.0, 1.0):
                    raise ValueError("invalid per-position KL or top1")
                total_kl += kl
                total_q += nq
                total_p += np
                agree += int(t1)
                count += 1
    if count != positions:
        raise ValueError("missing KLD rows")
    top1 = agree / count
    mean = total_kl / count
    p_ratio = math.exp((total_q - total_p) / count)
    reported = stats["kld"]["mean"]
    if abs(mean - _number(reported, "reported mean KLD")) > 1e-4:
        raise ValueError("reported aggregate KLD does not match .rows")
    if abs(top1 - _number(stats["top1_agreement"], "top1", hi=1)) > 1e-5:
        raise ValueError("reported top1 does not match .rows")
    if abs(p_ratio - _number(stats["ppl"]["ratio"], "ppl ratio")) > 1e-3:
        raise ValueError("reported PPL ratio does not match .rows")
    return {"positions": count, "actual_mean_kld": mean,
            "actual_top1_agreement": top1, "actual_ppl_ratio": p_ratio}


def gate(
    bundle_dir: Path, checkpoint: Path, reference_dir: Path,
    report_path: Path, candidate: Path, policy_path: Path,
    output: Path,
) -> dict:
    if not candidate.is_file() or not str(candidate).endswith(".rad"):
        raise ValueError("native candidate RAD2 model is required")
    if not report_path.is_file() or not reference_dir.is_dir():
        raise ValueError("missing Radiance KLD report or original BF16 reference")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("refuse to overwrite KLD gate receipt")
    policy = load_policy(policy_path)
    bundle, holdout_docs, sources = verify_corpus(bundle_dir, checkpoint)
    reference_path = reference_dir / "kld.json"
    ref = json.loads(reference_path.read_text())
    report = json.loads(report_path.read_text())
    holdout = (bundle_dir / "holdout.jsonl").resolve()
    if (Path(ref.get("corpus", "")).resolve() != holdout
            or Path(report.get("corpus", "")).resolve() != holdout
            or Path(report.get("ref_dir", "")).resolve() != reference_dir.resolve()):
        raise ValueError("reference or candidate scored the wrong/overlapping corpus")
    if ref.get("top") != 128 or report.get("ref_form") != "top-128 + tail":
        raise ValueError("only same sparse top-128+tail reference form is admitted")
    if not isinstance(ref.get("format"), str) or not ref["format"]:
        raise ValueError("missing native KLD reference format")
    if (not isinstance(ref.get("rows"), int) or ref["rows"] < 1
            or len(ref.get("docs", [])) != holdout_docs
            or report.get("docs") != holdout_docs
            or Path(report.get("reference", "")).resolve() != Path(ref.get("model", "")).resolve()
            or Path(report.get("model", "")).resolve() != candidate.resolve()
            or candidate.resolve() == Path(ref["model"]).resolve()):
        raise ValueError("different model/reference/doc count or KLD position identity")
    positions = ref["rows"]
    all_stats = report.get("all", {})
    if _number(all_stats.get("positions"), "all.positions") != positions:
        raise ValueError("report/reference scored-position count drift")
    holdout_lines = [
        json.loads(raw) for raw in (bundle_dir / "holdout.jsonl").read_text().splitlines()
    ]
    accumulated = 0
    reference_source_positions: Counter[str] = Counter()
    for row, expected in zip(ref["docs"], holdout_lines, strict=True):
        if not isinstance(row, dict):
            raise TypeError("invalid KLD reference document")
        if not isinstance(row.get("tokens"), int) or not isinstance(row.get("score_from"), int):
            raise TypeError("invalid native token metadata")
        tokens, from_token = row["tokens"], row["score_from"]
        if (tokens <= 0 or not 0 <= from_token < tokens
                or row.get("source") != expected["source"]
                or row.get("row0") != accumulated):
            raise ValueError("native KLD document order, source, or offsets drift")
        if from_token != expected.get("score_from", 0):
            raise ValueError("native KLD score_from differs from holdout request")
        n_scored = max(0, tokens - from_token - 1)
        reference_source_positions[row["source"]] += n_scored
        accumulated += n_scored
    if accumulated != positions:
        raise ValueError("native KLD scored-position census mismatch")
    source_summary = report.get("by_source", {})
    if not isinstance(source_summary, dict) or set(source_summary) != sources:
        raise ValueError("source-specific held-out coverage missing")
    for source, source_stats in source_summary.items():
        if (_number(source_stats.get("positions"), source + ".positions")
                != reference_source_positions[source]):
            raise ValueError(
                "per-source KLD scored positions disagree with reference documents"
            )
    row_check = validate_rows(
        Path(str(report_path) + ".rows"), positions, all_stats
    )
    failures = []
    def check(ok: bool, description: str) -> None:
        if not ok:
            failures.append(description)
    check(positions >= policy["min_positions"], "insufficient KLD positions")
    kl = all_stats["kld"]
    check(_number(kl.get("mean"), "KL.mean") <= policy["max_mean_kld"], "mean KL")
    check(_number(kl.get("p99"), "KL.p99") <= policy["max_p99_kld"], "p99 KL")
    check(_number(kl.get("p99_9"), "KL.p99_9") <= policy["max_p999_kld"], "p99.9 KL")
    check(_number(all_stats.get("top1_agreement"), "top1", hi=1)
          >= policy["min_top1"], "top1 agreement")
    check(_number(all_stats["ppl"].get("ratio"), "ppl.ratio")
          <= policy["max_ppl_ratio"], "perplexity ratio")
    for name, source in sorted(source_summary.items()):
        check(_number(source.get("positions"), name+".positions")
              >= policy["min_source_positions"], f"{name}: too few positions")
        check(_number(source.get("top1_agreement"), name+".top1", hi=1)
              >= policy["min_source_top1"], f"{name}: top1")
        check(_number(source["kld"].get("mean"), name+".mean")
              <= policy["max_source_mean_kld"], f"{name}: mean KL")
    receipt = {
        "schema": SCHEMA,
        "status": "PROVISIONAL_THRESHOLDS_PASS_NOT_PRODUCTION_APPROVAL"
                  if not failures else "QUALITY_THRESHOLD_REJECTED",
        "failures": failures,
        "quality_approved": False,
        "production_approved": False,
        "claim_scope": "Native sparse top-128+tail held-out KLD only; IQ6R "
                       "encoding, kernel ABI/dispatch, long-context serving, "
                       "resident KV and task accuracy remain separately gated.",
        "reference_manifest_sha256": sha_file(reference_path),
        "candidate_report_sha256": sha_file(report_path),
        "candidate_rows_sha256": sha_file(Path(str(report_path) + ".rows")),
        "candidate_model_sha256": sha_file(candidate),
        "corpus_manifest_sha256": sha_file(bundle_dir / "bundle.json"),
        "holdout_sha256": bundle["holdout_sha256"],
        "policy_sha256": sha_file(policy_path),
        "reference_form": report["ref_form"],
        "heldout_documents": holdout_docs,
        "metrics": row_check,
        "by_source": {key: {"positions": value["positions"],
                            "mean_kld": value["kld"]["mean"],
                            "top1_agreement": value["top1_agreement"]}
                      for key, value in sorted(source_summary.items())},
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("bundle", "checkpoint", "reference-dir", "report",
                  "candidate", "policy", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = gate(args.bundle, args.checkpoint, args.reference_dir,
                      args.report, args.candidate, args.policy, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if not result["failures"] else 3
    except (OSError, KeyError, ValueError, TypeError, OverflowError) as exc:
        print(f"VTP06 quality gate FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
