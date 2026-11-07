#!/usr/bin/env python3
"""Flash-Next performance parity: the before/after measurement of a Radiance pin bump (bump-radiance section 4).

flashnext-parity.sh drives the launches and writes raw files under OUT/launch-<i>/; this module does everything that is
not process control, so it can be tested without cards:

  run-info OUT --build DIR --launches N --port P --set NAME=VALUE...   OUT/run.json: build, revision, pin, date, workload
  warmup-plan MANIFEST                  {"policy": bool, "warmup_s": float, ...} from spec.handoff.placement.live_policy
  doc-calibrate --url U --tokens T --out FILE   chars per token of the docs corpus, measured with one probe request
  doc-prefill --url U --calibration FILE --slot measure:K|warmup:K --out FILE   one long document prefill, max_tokens 1
  policy-line LOG                       the helper's last `policy placement` line as JSON (committed, bytes_moved)
  summarise OUT                         OUT/parity.json: per metric the per-launch values and their median
  baseline OUT --name NAME [--to FILE]  a baseline from a measured parity.json, tolerances derived from the spread
  compare OUT --baseline FILE           table against a baseline; exit 1 on a gated regression beyond its allowance

Tokens/s is never gated across builds (MTP draft acceptance differs per build): the gated metrics are cold prefill and
engine step time; acceptance and the live policy's transactions are reported only. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = 1
BASELINE_SCHEMA = 1
DEFAULT_BASELINE = REPO / "docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json"
RECIPES = REPO / "engines/radiance/recipes.toml"

# name -> (direction, gated). direction: which way is better.
METRICS = {
    "cold_ttft_s": ("lower", True),
    "doc_prefill_tps": ("higher", True),
    "decode1_ms_per_step": ("lower", True),
    "decode6_ms_per_step": ("lower", True),
    "sess8_ttft_s": ("lower", True),
    "sess8_ms_per_step": ("lower", True),
    "long_prefill_tps": ("higher", True),
    "long_ms_per_step": ("lower", True),
    "decode1_acceptance": ("higher", False),
    "decode6_acceptance": ("higher", False),
    "long_acceptance": ("higher", False),
    "policy_transactions": ("higher", False),
}


class ParityError(RuntimeError):
    pass


def _load(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc


def _write(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------- run info

def git_revision(repo: Path = REPO) -> str:
    try:
        rev = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParityError(f"cannot read the checkout's git revision: {exc}") from exc
    return rev.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def radiance_pin(recipes: Path = RECIPES) -> dict:
    """The pin the repo builds radiance at: engines/radiance/recipes.toml `pinned` (tag) and `pinned-version`."""
    try:
        data = tomllib.loads(Path(recipes).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ParityError(f"cannot read the radiance pin from {recipes}: {exc}") from exc
    if "pinned" not in data:
        raise ParityError(f"{recipes} has no `pinned` key")
    return {"pinned": data["pinned"], "pinned_version": data.get("pinned-version")}


def build_identity(build: Path) -> dict:
    """The build's own record (bigcherry-build-metadata-radiance.json) when it has one: which build was measured."""
    meta = Path(build) / "bigcherry-build-metadata-radiance.json"
    if not meta.is_file():
        return {}
    data = _load(meta)
    return {k: data.get(k) for k in ("build_id", "build_plan_id", "source_slice_id", "binary_hash", "runtime_bundle_hash")}


def cmd_run_info(args) -> int:
    workload = {}
    for item in args.set or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise ParityError(f"--set wants NAME=VALUE, got {item!r}")
        workload[name] = int(value) if re.fullmatch(r"-?\d+", value) else value
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "run.json", {
        "schema": SCHEMA, "build": str(Path(args.build).resolve()), "build_identity": build_identity(Path(args.build)),
        "revision": git_revision(), "radiance_pin": radiance_pin(),
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "launches": args.launches, "port": args.port, "workload": workload})
    return 0


# ------------------------------------------------------------------------------------------------------------------ warmup

def warmup_plan(manifest: dict) -> dict:
    """With the live expert-placement policy on, wait until it has had (min_samples + 1) heat windows plus a second, so a
    measurement never starts before the policy could have moved anything. Without a policy there is nothing to wait for."""
    policy = (((manifest.get("spec") or {}).get("handoff") or {}).get("placement") or {}).get("live_policy")
    if not policy:
        return {"policy": False, "warmup_s": 0.0}
    cadence_ns, min_samples = int(policy["cadence_ns"]), int(policy["min_samples"])
    return {"policy": True, "cadence_ns": cadence_ns, "min_samples": min_samples,
            "warmup_s": (min_samples + 1) * cadence_ns / 1e9 + 1}


def cmd_warmup_plan(args) -> int:
    plan = warmup_plan(_load(Path(args.manifest)))
    plan["manifest"] = str(args.manifest)
    rendered = json.dumps(plan, sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


# ------------------------------------------------------------------------------------------------------- document prefill

def docs_corpus(repo: Path = REPO) -> str:
    """The repo's docs/**/*.md in path order, one text: the same checkout gives the same corpus."""
    files = sorted(p for p in (repo / "docs").rglob("*.md") if p.is_file())
    if not files:
        raise ParityError(f"no docs/**/*.md under {repo}")
    return "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def _completion(url: str, prompt: str, timeout: float) -> tuple[int, float]:
    """One non-streaming completion of max_tokens 1 at temperature 0: (server prompt_tokens, wall seconds)."""
    body = json.dumps({"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except (OSError, ValueError) as exc:
        raise ParityError(f"document prefill request failed: {exc}") from exc
    wall = time.perf_counter() - t0
    tokens = int((payload.get("usage") or {}).get("prompt_tokens") or 0)
    if tokens <= 0:
        raise ParityError("the server reported no usage.prompt_tokens for a document prefill")
    return tokens, wall


def probe_text(corpus: str, target_tokens: int) -> str:
    """The probe is the corpus' last `target_tokens` characters (at least one character a token, so it is never longer
    than a document prompt), a region no measured or warmup document prompt uses."""
    return corpus[-target_tokens:]


def doc_slices(corpus_len: int, chars: int, probe_chars: int, measured: int):
    """(measured offsets, warmup region start, warmup region length). Measured prompts are the first `measured` slots of
    the corpus; warmup prompts cycle through the rest before the probe region, so a measured prompt is never a prefix the
    server has seen."""
    usable = corpus_len - probe_chars
    warm_start = measured * chars
    warm_len = usable - warm_start
    if warm_len < 2 * chars:
        raise ParityError(f"the docs corpus ({corpus_len} chars) is too short for {measured} disjoint document prompts of "
                          f"{chars} chars, a warmup region and a probe")
    return [i * chars for i in range(measured)], warm_start, warm_len


def doc_offset(slot: str, chars: int, corpus_len: int, probe_chars: int, measured: int) -> int:
    kind, _, index = slot.partition(":")
    k = int(index)
    offsets, warm_start, warm_len = doc_slices(corpus_len, chars, probe_chars, measured)
    if kind == "measure":
        if not 0 <= k < measured:
            raise ParityError(f"measured slot {k} outside 0..{measured - 1}")
        return offsets[k]
    if kind == "warmup":
        span = warm_len - chars                      # the last start that still leaves a whole prompt in the region
        return warm_start + (k * chars) % (span + 1)
    raise ParityError(f"slot must be measure:K or warmup:K, got {slot!r}")


def cmd_doc_calibrate(args) -> int:
    corpus = docs_corpus()
    probe = probe_text(corpus, args.tokens)
    tokens, wall = _completion(args.url, probe, args.timeout)
    cpt = len(probe) / tokens
    chars = round(args.tokens * cpt)
    doc_slices(len(corpus), chars, len(probe), args.measured)        # refuse now, before any warmup, if it cannot fit
    _write(Path(args.out), {"target_tokens": args.tokens, "chars_per_token": cpt, "chars": chars,
                            "measured_slots": args.measured, "probe_chars": len(probe), "probe_prompt_tokens": tokens,
                            "probe_wall_s": wall, "corpus_chars": len(corpus),
                            "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest()})
    return 0


def cmd_doc_prefill(args) -> int:
    cal = _load(Path(args.calibration))
    corpus = docs_corpus()
    if hashlib.sha256(corpus.encode("utf-8")).hexdigest() != cal["corpus_sha256"]:
        raise ParityError("the docs corpus changed since the calibration of this launch")
    offset = doc_offset(args.slot, cal["chars"], len(corpus), cal["probe_chars"], cal["measured_slots"])
    prompt = corpus[offset:offset + cal["chars"]]
    tokens, wall = _completion(args.url, prompt, args.timeout)
    _write(Path(args.out), {"slot": args.slot, "offset": offset, "chars": len(prompt), "target_tokens": cal["target_tokens"],
                            "prompt_tokens": tokens, "wall_s": wall, "prefill_tps": tokens / wall,
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return 0


# ------------------------------------------------------------------------------------------------------------ policy line

_POLICY = re.compile(r"policy placement helper=(\d+)\b.*?\bcommitted=(\d+)\b.*?\bbytes_moved=(\d+)\b")


def policy_line(text: str) -> dict:
    """The last `policy placement` line of a helper log: its counters are cumulative for the helper's life."""
    last = None
    for line in text.splitlines():
        m = _POLICY.search(line)
        if m:
            last = (line.strip(), m)
    if last is None:
        return {"found": False}
    line, m = last
    return {"found": True, "helper": int(m.group(1)), "committed": int(m.group(2)), "bytes_moved": int(m.group(3)),
            "line": line}


def cmd_policy_line(args) -> int:
    text = Path(args.log).read_text(encoding="utf-8", errors="replace") if Path(args.log).is_file() else ""
    rendered = json.dumps(policy_line(text), sort_keys=True)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


# --------------------------------------------------------------------------------------------------------------- summarise

def _rounds(path: Path) -> list:
    rec = _load(path)
    if not rec.get("ok"):
        raise ParityError(f"{path}: a stream failed (see its per_stream errors)")
    rounds = rec.get("rounds") or []
    if not rounds:
        raise ParityError(f"{path}: no rounds")
    return rounds


def _engine(rnd: dict, key: str, path: Path) -> float:
    eng = rnd.get("engine") or {}
    if key not in eng:
        raise ParityError(f"{path}: round {rnd.get('rep')} has no engine {key} (does the server expose /metrics?)")
    return float(eng[key])


def _files(launch: Path, pattern: str, expect: int | None = None) -> list:
    files = sorted(launch.glob(pattern))
    if not files or (expect is not None and len(files) != expect):
        raise ParityError(f"{launch}: expected {expect or 'some'} {pattern}, found {len(files)}")
    return files


def launch_metrics(launch: Path, workload: dict) -> tuple[dict, dict, dict]:
    """(metric -> per-launch value, metric -> repeat samples within the launch, detail) from one launch's raw files."""
    value, samples, detail = {}, {}, {}

    fills = [r["per_stream"][0]["ttft_ms"] / 1000.0
             for f in _files(launch, "cold-fill-*.json", workload.get("COLD_FILLS")) for r in _rounds(f)]
    value["cold_ttft_s"], samples["cold_ttft_s"] = statistics.fmean(fills), fills

    docs = [_load(f)["prefill_tps"] for f in _files(launch, "doc-prefill-*.json", workload.get("DOC_FILLS"))]
    value["doc_prefill_tps"], samples["doc_prefill_tps"] = statistics.fmean(docs), docs

    for name, file in (("decode1", "decode1.json"), ("decode6", "decode6.json")):
        rounds = _rounds(launch / file)
        steps = [_engine(r, "decode_ms_per_step", launch / file) for r in rounds]
        acc = [_engine(r, "draft_acceptance", launch / file) for r in rounds]
        value[f"{name}_ms_per_step"], samples[f"{name}_ms_per_step"] = statistics.fmean(steps), steps
        value[f"{name}_acceptance"], samples[f"{name}_acceptance"] = statistics.fmean(acc), acc
        detail[f"{name}_tokens_per_step"] = [_engine(r, "decode_tokens_per_step", launch / file) for r in rounds]
        detail[f"{name}_ttft_s"] = [r["aggregate"]["ttft_ms_max"] / 1000.0 for r in rounds]

    rounds = _rounds(launch / "sess8.json")
    cold = [r for r in rounds if r.get("rep") == 0]
    if not cold:
        raise ParityError(f"{launch}/sess8.json: no first (cold) round")
    value["sess8_ttft_s"] = cold[0]["aggregate"]["ttft_ms_p50"] / 1000.0
    detail["sess8_ttft_p50_s"] = [r["aggregate"]["ttft_ms_p50"] / 1000.0 for r in rounds]
    steps = [_engine(r, "decode_ms_per_step", launch / "sess8.json") for r in rounds]
    value["sess8_ms_per_step"], samples["sess8_ms_per_step"] = statistics.fmean(steps), steps

    prefill, steps, acc, prompt = [], [], [], []
    for f in _files(launch, "long-*.json", workload.get("LONG_SEEDS_COUNT")):
        for r in _rounds(f):
            steps.append(_engine(r, "decode_ms_per_step", f))
            acc.append(_engine(r, "draft_acceptance", f))
            s = r["per_stream"][0]
            prompt.append(s["prompt_tokens"])
            if r.get("rep") == 0:                       # a repeat of the same prompt is a prefix-cache hit, not a prefill
                prefill.append(s["prompt_tokens"] / (s["ttft_ms"] / 1000.0))
    if not prefill:
        raise ParityError(f"{launch}: no cold long-context round")
    value["long_prefill_tps"], samples["long_prefill_tps"] = statistics.fmean(prefill), prefill
    value["long_ms_per_step"], samples["long_ms_per_step"] = statistics.fmean(steps), steps
    value["long_acceptance"], samples["long_acceptance"] = statistics.fmean(acc), acc
    detail["long_prompt_tokens"] = prompt

    pol = _load(launch / "policy.json") if (launch / "policy.json").is_file() else {"found": False}
    value["policy_transactions"] = pol["committed"] if pol.get("found") else None
    detail["policy_bytes_moved"] = pol.get("bytes_moved")
    return value, samples, detail


def _median(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def summarise(out: Path) -> dict:
    out = Path(out)
    run = _load(out / "run.json")
    launches = sorted((p for p in out.glob("launch-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    if not launches:
        raise ParityError(f"{out}: no launch-* directories")
    per_launch, metrics = [], {name: {"direction": d, "gated": g, "launches": [], "samples": []}
                               for name, (d, g) in METRICS.items()}
    for launch in launches:
        info = _load(launch / "launch.json")
        value, samples, detail = launch_metrics(launch, run.get("workload") or {})
        per_launch.append({"launch": launch.name, "policy": info["policy"], "warmup_s": info["warmup_s"],
                           "detail": detail})
        for name in METRICS:
            metrics[name]["launches"].append(value.get(name))
            metrics[name]["samples"].append(samples.get(name, []))
    for m in metrics.values():
        m["median"] = _median(m["launches"])
    policies = {p["policy"] for p in per_launch}
    if len(policies) != 1:
        raise ParityError(f"{out}: the launches disagree on whether the live policy was on")
    return {"schema": SCHEMA, "kind": "flashnext-parity", "build": run["build"], "build_identity": run.get("build_identity"),
            "revision": run["revision"], "radiance_pin": run["radiance_pin"], "date": run["date"],
            "workload": run.get("workload") or {}, "policy": policies.pop(),
            "warmup_s": [p["warmup_s"] for p in per_launch], "launches": per_launch, "metrics": metrics}


def cmd_summarise(args) -> int:
    parity = summarise(Path(args.out))
    _write(Path(args.out) / "parity.json", parity)
    for name, m in parity["metrics"].items():
        vals = ", ".join("-" if v is None else f"{v:.4g}" for v in m["launches"])
        med = "-" if m["median"] is None else f"{m['median']:.4g}"
        print(f"{name:22s} median {med:>10s}  launches [{vals}]  ({m['direction']} better"
              f"{'' if m['gated'] else ', informational'})")
    return 0


# ---------------------------------------------------------------------------------------------------------------- baseline

def _rel_spread(values) -> float:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    centre = statistics.fmean(vals)
    return (max(vals) - min(vals)) / abs(centre) if centre else 0.0


def _parity_file(path: Path) -> Path:
    path = Path(path)
    return path / "parity.json" if path.is_dir() else path


def make_baseline(parity: dict, name: str, source: str) -> dict:
    """Per gated metric: the median as the value; the tolerance is the larger of the relative spread between launches and
    the largest relative spread of the repeats within one launch. Nothing is picked by hand, so one launch is refused."""
    count = len(parity["launches"])
    if count < 2:
        raise ParityError(f"a baseline needs at least two launches to measure their spread; {source} has {count}")
    metrics = {}
    for metric, m in parity["metrics"].items():
        entry = {"direction": m["direction"], "gated": m["gated"], "value": m["median"]}
        if m["gated"]:
            if m["median"] is None or any(v is None for v in m["launches"]):
                raise ParityError(f"{source}: gated metric {metric} is missing in a launch")
            across = _rel_spread(m["launches"])
            within = max((_rel_spread(s) for s in m["samples"]), default=0.0)
            entry.update({"tolerance": max(across, within), "spread_across_launches": across,
                          "spread_within_launch": within, "launches": m["launches"]})
        metrics[metric] = entry
    return {"schema": BASELINE_SCHEMA, "kind": "flashnext-parity-baseline", "name": name, "source_run": source,
            "build": parity["build"], "build_identity": parity.get("build_identity"), "revision": parity["revision"],
            "radiance_pin": parity["radiance_pin"], "date": parity["date"], "policy": parity["policy"],
            "launches": count, "workload": parity["workload"], "metrics": metrics}


def cmd_baseline(args) -> int:
    src = _parity_file(Path(args.out))
    baseline = make_baseline(_load(src), args.name, str(src.parent.resolve()))
    to = Path(args.to) if args.to else DEFAULT_BASELINE
    to.parent.mkdir(parents=True, exist_ok=True)
    _write(to, baseline)
    print(f"baseline {args.name}: {to}")
    for metric, e in baseline["metrics"].items():
        if e["gated"]:
            print(f"  {metric:22s} {e['value']:.4g}  tolerance {e['tolerance'] * 100:.2f}%")
    return 0


# ----------------------------------------------------------------------------------------------------------------- compare

def compare(parity: dict, baseline: dict) -> tuple[list, bool]:
    """Rows (metric, baseline, now, change %, allowance %, verdict) and whether any gated metric regressed beyond its
    allowance. One-sided: a change in the better direction is never a failure. A gated metric without a value now fails."""
    if parity.get("workload") != baseline.get("workload"):
        raise ParityError("the run's workload definition differs from the baseline's: they do not measure the same thing")
    rows, failed = [], False
    for metric, base in baseline["metrics"].items():
        now = (parity["metrics"].get(metric) or {}).get("median")
        ref = base["value"]
        change = (now - ref) / ref * 100.0 if now is not None and ref else None
        if not base["gated"]:
            rows.append((metric, ref, now, change, None, "info"))
            continue
        allowance = base["tolerance"] * 100.0
        if now is None:
            verdict = "MISSING"
        else:
            worse = change if base["direction"] == "lower" else -change
            verdict = "REGRESSION" if worse > allowance else ("better" if worse < 0 else "ok")
        failed |= verdict in ("REGRESSION", "MISSING")
        rows.append((metric, ref, now, change, allowance, verdict))
    return rows, failed


def render(rows) -> str:
    def num(v, fmt):
        return "-" if v is None else format(v, fmt)
    lines = [f"{'metric':22s} {'baseline':>12s} {'now':>12s} {'change %':>9s} {'allowance':>10s}  verdict"]
    for metric, ref, now, change, allowance, verdict in rows:
        lines.append(f"{metric:22s} {num(ref, '.4g'):>12s} {num(now, '.4g'):>12s} {num(change, '+.2f'):>9s} "
                     f"{('±' + num(allowance, '.2f')) if allowance is not None else '-':>10s}  {verdict}")
    return "\n".join(lines)


def cmd_compare(args) -> int:
    parity = _load(_parity_file(Path(args.out)))
    baseline = _load(Path(args.baseline))
    rows, failed = compare(parity, baseline)
    print(f"baseline {baseline.get('name')} (pin {baseline['radiance_pin'].get('pinned')}, rev {baseline['revision'][:12]}) "
          f"-> now (pin {parity['radiance_pin'].get('pinned')}, rev {parity['revision'][:12]})")
    if parity.get("policy") != baseline.get("policy"):
        print(f"note: live policy {'on' if parity.get('policy') else 'off'} now, "
              f"{'on' if baseline.get('policy') else 'off'} in the baseline")
    print(render(rows))
    print("FAIL: a gated metric regressed beyond its allowance (investigate; do not waive)" if failed else "PASS")
    return 1 if failed else 0


# -------------------------------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run-info")
    p.add_argument("out")
    p.add_argument("--build", required=True)
    p.add_argument("--launches", type=int, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--set", action="append", help="a workload constant NAME=VALUE (repeat)")
    p.set_defaults(fn=cmd_run_info)
    p = sub.add_parser("warmup-plan")
    p.add_argument("manifest")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_warmup_plan)
    p = sub.add_parser("doc-calibrate")
    p.add_argument("--url", required=True)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--measured", type=int, required=True, help="number of measured document prompts")
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_doc_calibrate)
    p = sub.add_parser("doc-prefill")
    p.add_argument("--url", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--slot", required=True, help="measure:K or warmup:K")
    p.add_argument("--timeout", type=float, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_doc_prefill)
    p = sub.add_parser("policy-line")
    p.add_argument("log")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_policy_line)
    p = sub.add_parser("summarise")
    p.add_argument("out")
    p.set_defaults(fn=cmd_summarise)
    p = sub.add_parser("baseline")
    p.add_argument("out", help="a measure OUT directory or its parity.json")
    p.add_argument("--name", required=True)
    p.add_argument("--to", help=f"default {DEFAULT_BASELINE.relative_to(REPO)}")
    p.set_defaults(fn=cmd_baseline)
    p = sub.add_parser("compare")
    p.add_argument("out", help="a measure OUT directory or its parity.json")
    p.add_argument("--baseline", required=True)
    p.set_defaults(fn=cmd_compare)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ParityError, KeyError, ValueError) as exc:
        print(f"flashnext-parity: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
