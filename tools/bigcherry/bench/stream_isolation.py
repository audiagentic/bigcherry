"""Isolation probe (RAD14): do the same prompts give the same replies alone and together?

Every prompt runs ALONE first (twice, to show the baseline is itself repeatable), then the same prompts run
simultaneously in groups of 2, 3, 4 ... (in order and reversed). At temperature 0 a correct server returns the same text
whatever else is in the batch, so a stream that differs from its own solo reply is a stream the batch changed. The record
gives, per stream, whether it matched, the length of the common prefix (where it first diverged), and the draft acceptance
of the round, so a collapse can be tied to the group size that causes it.
"""

from __future__ import annotations

import threading
import time

from bigcherry.bench import stream_bench as sb


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _group(url, model, prompts, order, prompt_tokens, max_tokens, seed, timeout, stagger_s=0.0):
    """The prompts `order` (a list of prompt indices) released together, or `stagger_s` seconds apart; returns (results,
    engine record)."""
    results: list = [None] * len(order)
    gate = threading.Barrier(len(order) + 1)
    before = sb.scrape_metrics(url)

    def work(slot, idx):
        gate.wait()
        if stagger_s > 0:
            time.sleep(slot * stagger_s)
        results[slot] = sb.one_stream(url, model, idx, prompts[idx], max_tokens, 0.0, timeout, seed)

    threads = [threading.Thread(target=work, args=(slot, idx), daemon=True) for slot, idx in enumerate(order)]
    for t in threads:
        t.start()
    gate.wait()
    for t in threads:
        t.join()
    return results, sb.engine_delta(before, sb.scrape_metrics(url), None)


def run(url: str, model: str, n_prompts: int, group_sizes: list, prompt_tokens: int, max_tokens: int,
        seed: int = 1, timeout: float = 600.0, stagger_s: float = 0.0) -> dict:
    if n_prompts < 2 or not group_sizes or any(n < 2 or n > n_prompts for n in group_sizes):
        raise sb.StreamBenchError("isolation needs at least 2 prompts and group sizes within 2..prompts")
    prompts = [sb.make_prompt(i, prompt_tokens, seed, False) for i in range(n_prompts)]
    solo, solo_repeat = {}, {}
    for i in range(n_prompts):
        a, _ = _group(url, model, prompts, [i], prompt_tokens, max_tokens, seed, timeout)
        b, _ = _group(url, model, prompts, [i], prompt_tokens, max_tokens, seed, timeout)
        if not (a[0].ok and b[0].ok):
            raise sb.StreamBenchError(f"solo run of prompt {i} failed: {a[0].error or b[0].error}")
        solo[i], solo_repeat[i] = a[0].text, b[0].text
    record = {"schema": 1, "config": {"url": url, "n_prompts": n_prompts, "group_sizes": group_sizes,
                                       "prompt_tokens": prompt_tokens, "max_tokens": max_tokens, "seed": seed, "stagger_s": stagger_s},
              "solo": [{"prompt": i, "chars": len(solo[i]), "repeat_match": solo[i] == solo_repeat[i],
                        "repeat_prefix": _common_prefix(solo[i], solo_repeat[i]), "head": solo[i][:200]}
                       for i in range(n_prompts)],
              "groups": []}
    for n in group_sizes:
        for label, order in (("forward", list(range(n))), ("reversed", list(reversed(range(n))))):
            results, engine = _group(url, model, prompts, order, prompt_tokens, max_tokens, seed, timeout, stagger_s)
            streams = []
            for slot, idx in enumerate(order):
                r = results[slot]
                streams.append({"slot": slot, "prompt": idx, "ok": r.ok, "error": r.error,
                                "match": bool(r.ok and r.text == solo[idx]),
                                "common_prefix": _common_prefix(r.text, solo[idx]) if r.ok else 0,
                                "chars": len(r.text), "solo_chars": len(solo[idx]),
                                "tail_at_divergence": r.text[_common_prefix(r.text, solo[idx]):][:120] if r.ok else "",
                                "solo_at_divergence": solo[idx][_common_prefix(r.text, solo[idx]):][:120] if r.ok else ""})
            record["groups"].append({"size": n, "order": label, "engine": engine, "streams": streams})
    return record


def flat_metrics(record: dict) -> dict:
    """iso_solo_repeat_frac (solo replies that repeat), then per group size iso_nN_match_frac (streams equal to their solo
    reply), iso_nN_prefix_min (shortest common prefix, in characters) and iso_nN_acceptance (mean draft acceptance)."""
    out = {"iso_solo_repeat_frac": sum(1 for s in record["solo"] if s["repeat_match"]) / len(record["solo"])}
    for n in record["config"]["group_sizes"]:
        gs = [g for g in record["groups"] if g["size"] == n]
        streams = [s for g in gs for s in g["streams"]]
        out[f"iso_n{n}_match_frac"] = sum(1 for s in streams if s["match"]) / len(streams)
        out[f"iso_n{n}_prefix_min"] = float(min(s["common_prefix"] for s in streams))
        acc = [g["engine"]["draft_acceptance"] for g in gs if g["engine"] and "draft_acceptance" in g["engine"]]
        if acc:
            out[f"iso_n{n}_acceptance"] = sum(acc) / len(acc)
    return out
