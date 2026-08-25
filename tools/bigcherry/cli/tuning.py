"""CLI presentation handlers for tuning and replay workflows."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from .. import paths, replay_inspect


def cmd_generate(args: Namespace) -> int:
    from .. import __main__ as legacy, autotune_catalog

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
