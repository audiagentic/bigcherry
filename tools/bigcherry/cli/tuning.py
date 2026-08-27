"""CLI presentation handlers for tuning and replay workflows."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from ..core import paths
from ..tuning import replay_inspect


def cmd_generate(args: Namespace) -> int:
    from .. import __main__ as legacy
    from ..tuning import catalog as autotune_catalog

    root = paths.llama_root(args.llama_root)
    record = legacy._record_for(root)
    if not args.force and record.stage not in (
        "patched",
        "generated",
        "built",
        "tested",
        "tuned",
        "validated",
    ):
        print(
            "refusing to generate against an unpatched tree.\n"
            "  run `python -m bigcherry apply` first, or pass --force.",
            file=sys.stderr,
        )
        return 2

    forwarded = ["--variant-set", args.variant_set, "--arch", args.arch]
    if args.llama_root:
        forwarded += ["--llama-root", args.llama_root]
    if args.inventory:
        forwarded += ["--inventory", args.inventory]
    if args.winners:
        forwarded += ["--winners", args.winners]
    if args.generated_root:
        forwarded += ["--generated-root", args.generated_root]
    if args.dry_run:
        forwarded += ["--dry-run"]

    status = autotune_catalog.main(forwarded)
    if status == 0 and not args.dry_run:
        manifest_path = (
            paths.artifact_dir(record.revision) / "hip-autotune-manifest.json"
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record.manifest_hash = manifest["manifest_hash"]
        record.advance_to("generated")
        record.save()
    return status


def cmd_replay_inspect(args: Namespace) -> int:
    """Inspect the registry and replay cache through the real C++ loader."""
    tool = replay_inspect.find_tool(args.tool)
    report = replay_inspect.run_tool(
        tool,
        cache=Path(args.cache) if args.cache else None,
        interpreter=args.tool_interpreter or None,
    )
    if args.manifest:
        report["manifest"] = replay_inspect.manifest_agreement(
            report, Path(args.manifest)
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(replay_inspect.format_report(report))
    return report["_exit"]


def cmd_project_replay(args: Namespace) -> int:
    """HI121 M4: project a measurements JSONL to the rows a specific target
    HIP build can safely reuse, using its own verified producer-capability
    provenance -- never touches replay.py's own reader/writer, wire format,
    or the production C++ resolver. See tuning.replay_projection."""
    from ..tuning import replay_projection

    try:
        summary = replay_projection.project_measurements(
            Path(args.measurements),
            Path(args.output),
            dispatch_db=Path(args.dispatch_db),
            source_build_id=args.source_build_id,
            source_manifest_path=Path(args.source_manifest),
            target_manifest_path=Path(args.target_manifest),
            vendor_root=Path(args.vendor_root),
        )
    except (replay_projection.ProjectionError, OSError, ValueError, KeyError) as exc:
        print(f"project-replay: {exc}", file=sys.stderr)
        return 1

    report = {
        "examined": summary.examined,
        "retained": summary.retained,
        "omitted_missing_producer_capability": summary.omitted_missing_producer_capability,
        "omitted_missing_target_capability": summary.omitted_missing_target_capability,
        "omitted_unsupported_domain": summary.omitted_unsupported_domain,
        "omitted_unverified_source": summary.omitted_unverified_source,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"examined {report['examined']}, retained {report['retained']} -- "
            f"omitted: {report['omitted_missing_producer_capability']} missing-producer-capability, "
            f"{report['omitted_missing_target_capability']} missing-target-capability, "
            f"{report['omitted_unsupported_domain']} unsupported-domain, "
            f"{report['omitted_unverified_source']} unverified-source"
        )
    return 0


def cmd_reattest(args: Namespace) -> int:
    """HI121 close-out step 7 (HI128): re-verify an existing schema-8
    winner's ORIGINAL measurements/manifest against HI127's strengthened-
    ingest profile and attest it if it genuinely passes. See
    tuning.reattest for the full trust argument -- this never replays
    load_measurements() (which would mutate the evidence under
    examination) and always requires a real, compiled HI125 verifier
    binary, even in --dry-run."""
    from ..tuning import reattest as reattest_module
    from ..tuning import signature_digest_verification as sdv

    verifier = sdv.make_signature_digest_verifier(
        binary=Path(args.signature_verifier_binary),
        vendor_root=Path(args.signature_verifier_vendor_root),
        seed=args.seed,
    )
    try:
        report = reattest_module.reattest_winners(
            Path(args.database),
            source_build_id=args.source_build_id,
            measurements_path=Path(args.measurements),
            manifest_path=Path(args.manifest),
            signature_digest_verifier=verifier,
            signature_source_paths=[Path(p) for p in args.signature_source],
            dry_run=args.dry_run,
        )
    except reattest_module.ReattestationError as exc:
        print(f"reattest: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(
            {
                "examined": report.examined,
                "attested": report.attested,
                "already_attested": report.already_attested,
                "backfilled_build_descriptor": report.backfilled_build_descriptor,
                "backfilled_build_capability": report.backfilled_build_capability,
                "outcomes": [
                    {"dispatch": o.dispatch, "status": o.status, "detail": o.detail}
                    for o in report.outcomes
                ],
            },
            indent=2, sort_keys=True,
        ))
    else:
        print(
            f"examined {report.examined}, attested {report.attested} "
            f"({report.already_attested} already attested)"
            + (" [dry-run]" if args.dry_run else "")
        )
        if report.backfilled_build_descriptor:
            print("  backfilled build_descriptor_hash")
        if report.backfilled_build_capability:
            print("  backfilled build_capability")
        from collections import Counter
        counts = Counter(o.status for o in report.outcomes)
        for status, count in sorted(counts.items()):
            if status not in ("attested", "already_attested", "would_attest"):
                print(f"  {count} {status}")
    return 0


def cmd_inventory(args: Namespace, *, subcmd: str) -> int:
    """Dispatch to inventory record/tuning subcommand."""
    from ..tuning import inventory as inv_mod

    if subcmd == "record":
        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"no such record file: {record_path}", file=sys.stderr)
            return 2
        try:
            record = inv_mod.read_jsonl(record_path)
        except inv_mod.RecordError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        inventory = inv_mod.build_inventory(record)
        inventory_path = (
            Path(args.inventory)
            if args.inventory
            else record_path.with_suffix(".inventory.json")
        )
        inventory_path.write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8", newline=""
        )

        database_path = (
            Path(args.database) if args.database else record_path.with_suffix(".sqlite")
        )
        counts = inv_mod.build_database(
            record, database_path, paths.SQL / "dispatch-db.sql"
        )

        print(f"read {len(record.observations)} observation(s) from {record_path}")
        print(f"  types: mmq={inventory['mmq_types']} mmvq={inventory['mmvq_types']}")
        print(f"         mmvf={inventory['mmvf_types']} mmf={inventory['mmf_types']}")
        print(f"  widths: {inventory['widths']}")
        print(f"  blas observed: {inventory['uses_blas']}")
        print(f"  inventory: {inventory_path}")
        print(
            f"  database:  {database_path} "
            f"({counts['signatures']} signatures, {counts['hardware']} hardware)"
        )
        return 0

    elif subcmd == "tuning":
        meas_path = Path(args.measurements)
        if not meas_path.is_file():
            print(f"no such measurements file: {meas_path}", file=sys.stderr)
            return 2

        db_path = (
            Path(args.database) if args.database else meas_path.with_suffix(".sqlite")
        )
        manifest_path = Path(args.manifest) if args.manifest else None

        # HI125 close-out step 6: --signature-verifier-binary/-vendor-root
        # are an all-or-none pair, and require --manifest too -- without a
        # manifest, build_attested is always False (HI127's own gate) and
        # this expensive real C++ verification would create zero
        # winner_verification attestations, silently wasting an operator's
        # GPU time on nothing.
        verifier_binary = args.signature_verifier_binary
        verifier_vendor_root = args.signature_verifier_vendor_root
        if bool(verifier_binary) != bool(verifier_vendor_root):
            print(
                "inventory tuning: --signature-verifier-binary and "
                "--signature-verifier-vendor-root must be supplied together",
                file=sys.stderr,
            )
            return 2
        signature_digest_verifier = None
        if verifier_binary:
            if manifest_path is None:
                print(
                    "inventory tuning: --manifest is required when a signature "
                    "verifier is supplied -- otherwise no winner can be attested",
                    file=sys.stderr,
                )
                return 2
            from ..tuning import signature_digest_verification as sdv
            signature_digest_verifier = sdv.make_signature_digest_verifier(
                binary=Path(verifier_binary),
                vendor_root=Path(verifier_vendor_root),
                seed=args.signature_verifier_seed,
            )

        try:
            counts = inv_mod.load_measurements(
                meas_path,
                db_path,
                paths.SQL / "dispatch-db.sql",
                manifest_path=manifest_path,
                signature_source_paths=[Path(p) for p in args.signature_source],
                signature_digest_verifier=signature_digest_verifier,
                # adversarial-review follow-up (2026-08-27): an operator who
                # asked for real C++ verification must never silently get
                # an unattested load instead -- e.g. a --manifest path that
                # does not actually exist would otherwise be treated as "no
                # manifest supplied" and quietly commit zero attestations.
                require_strengthened_ingest=signature_digest_verifier is not None,
            )
        except inv_mod.RecordError as exc:
            print(f"inventory tuning: {exc}", file=sys.stderr)
            return 1

        print(
            f"loaded {counts['results']} result(s) with "
            f"{counts['measurements']} measurement(s) and "
            f"{counts['candidates']} candidate(s) into {db_path}"
        )
        return 0

    elif subcmd == "hot-list":
        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"no such record file: {record_path}", file=sys.stderr)
            return 2
        try:
            record = inv_mod.read_jsonl(record_path)
        except inv_mod.RecordError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        output = Path(args.output) if args.output else record_path.with_suffix(".hot")
        measurements = Path(args.measurements) if args.measurements else None
        summary = inv_mod.write_hot_list(record, output, measurements=measurements)

        print(f"wrote {output}")
        print(f"  {summary['signatures']} signature(s), basis {summary['basis']}")
        for row in summary["rows"][:5]:
            print(
                f"  {row['rank']:>3}  {row['signature'][:16]}  "
                f"{row['calls']:>8} calls  {row['share_pct']:6.2f}%  "
                f"cum {row['cum_share_pct']:6.2f}%"
            )
        return 0

    elif subcmd == "workload-check":
        from ..analysis import report as report_mod

        record_path = Path(args.record)
        if not record_path.is_file():
            print(f"no such record file: {record_path}", file=sys.stderr)
            return 2
        try:
            record = inv_mod.read_jsonl(record_path)
        except inv_mod.RecordError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if args.measurements:
            measurements_path = Path(args.measurements)
            if not measurements_path.is_file():
                print(
                    f"no such measurements file: {measurements_path}", file=sys.stderr
                )
                return 2
            tuned_signatures = {
                row["signature"]
                for row in report_mod.read_measurements_jsonl(measurements_path)
                if row.get("signature")
            }
            tuned_source = str(measurements_path)
        else:
            from ..tuning import replay as replay_cache

            cache_path = Path(args.cache)
            if not cache_path.is_file():
                print(f"no such cache file: {cache_path}", file=sys.stderr)
                return 2
            # read_cache()/validate_blob() fail closed with SystemExit on
            # truncation, bad magic/checksum, v4 input, or unrecognised
            # match_kind; translate that into a normal CLI error here rather
            # than letting a library exception escape.
            try:
                cache_header, cache_entries = replay_cache.read_cache(
                    cache_path.read_bytes()
                )
            except SystemExit as exc:
                print(f"cannot read replay cache {cache_path}: {exc}", file=sys.stderr)
                return 1
            # A valid zero-entry cache is not an error: it reports zero
            # coverage, which is the honest answer for a cache with no
            # winners.
            tuned_signatures = {entry["signature"] for entry in cache_entries}
            tuned_source = (
                f"{cache_path} (v{cache_header['version']} replay cache, "
                f"{len(cache_entries)} entries)"
            )

        record_digest = inv_mod.workload_digest(
            o["signature"] for o in record.observations if o.get("signature")
        )
        tuned_digest = inv_mod.workload_digest(tuned_signatures)
        overlap = inv_mod.workload_overlap(record, tuned_signatures)

        print(f"workload  {record_digest}  ({record_path})")
        print(f"tuned     {tuned_digest}  ({tuned_source})", end="")
        print("  DIFFERENT" if record_digest != tuned_digest else "  SAME")
        print()
        print(
            f"coverage  {overlap['signatures_covered']} of "
            f"{overlap['signatures_observed']} observed signatures have a tuned winner"
        )
        print(
            f"          {overlap['calls_covered']} of {overlap['calls_observed']} "
            f"calls covered ({overlap['covered_share_pct']:.1f}%)"
        )
        print()
        print(
            "Advisory only: this does not gate any cache load or promotion "
            "decision -- it says the tune was measured for a different "
            "workload, not that it is unsafe."
        )
        return 0

    else:
        # Backward compat: positional arg means record mode (no subcommand)
        return cmd_inventory(args, subcmd="record")


def cmd_tune_campaign(args: Namespace) -> int:
    """HI130: the full record->tune->correctness->promote->replay pipeline
    as one command. See tuning/workflow.py for the actual orchestration --
    this handler only resolves CLI-level context/config and renders the
    result, mirroring cli/build.py::cmd_build_new's own shape.
    """
    from ..core import config as campaign_config
    from ..core.artifacts import ArtifactStore
    from ..core.context import ProjectContext
    from ..tuning import workflow
    from dataclasses import asdict

    context = ProjectContext.resolve(
        work_root=None, upstream_repo=Path(args.llama_root) if args.llama_root else None
    )
    try:
        cfg = campaign_config.load(context.config_path)
    except campaign_config.ConfigError as exc:
        print(f"tune-campaign: {exc}", file=sys.stderr)
        return 2

    if args.runtime_profile not in cfg.runtime_profiles:
        print(
            f"tune-campaign: no runtime-profile named {args.runtime_profile!r} -- "
            f"known: {sorted(cfg.runtime_profiles)}",
            file=sys.stderr,
        )
        return 2

    store = ArtifactStore(context.work_root / "artifacts-store")
    try:
        receipt = workflow.run_tune_campaign(
            context=context,
            cfg=cfg,
            store=store,
            model_path=Path(args.model),
            platform_name=args.platform,
            devices=args.devices,
            runtime_profile_name=args.runtime_profile,
            source_name=args.source,
            run_id=args.run_id,
            workdir=Path(args.workdir) if args.workdir else None,
            tune_screen_samples=args.tune_screen_samples,
            tune_final_samples=args.tune_final_samples,
            correctness_seeds=tuple(int(s) for s in args.correctness_seeds.split(",")),
            promotion_q=args.q,
            promotion_threshold_pct=args.threshold_pct,
            promotion_resamples=args.resamples,
        )
    except workflow.TuneCampaignError as exc:
        print(f"tune-campaign: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(receipt), indent=2, sort_keys=True))
    else:
        print(f"campaign_run_id: {receipt.campaign_run_id}")
        print(f"promoted: {receipt.promoted_before_evidence} -> {receipt.promoted_after_evidence}")
        if receipt.replay_coverage is not None:
            print(f"replay coverage: {receipt.replay_coverage}")
        print(
            f"receipt: {context.work_root / 'tune-campaigns' / receipt.campaign_run_id / 'tune-campaign-receipt.json'}"
        )
    return 0
