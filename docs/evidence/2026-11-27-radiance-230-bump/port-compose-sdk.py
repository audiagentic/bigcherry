#!/usr/bin/env python3
"""Historical source port: use the named recipe SDK, not arbitrary system clang."""
from pathlib import Path
import argparse
a=argparse.ArgumentParser()
a.add_argument("--apply",action="store_true",required=True)
a.parse_args()
p=Path(__file__).resolve().parents[3]
f=p/"tools/bigcherry/patch/compose_check.py"
s=f.read_text()
start=s.index("def compiler()")
end=s.index("\n\ndef gate1",start)
s=s[:start]+'''def compiler(engine: str) -> Path | None:
    """Find the same named ROCm SDK as the recipe, never a generic PATH clang.

    An explicit ROCM_PATH selects the SDK for an already-resolved build environment.
    For a standalone check, a unique SDK root from the engine's platforms suffices.
    Missing or ambiguous SDK configuration is a skipped compile, not a fallback.
    """
    base = os.environ.get("ROCM_PATH")
    if not base:
        from ..core import config
        from ..core.context import ProjectContext
        try:
            recipes = config.load(ProjectContext.resolve(engine=engine).config_path)
        except (OSError, ValueError):
            return None
        roots = {platform.sdk_root for platform in recipes.platforms.values()
                 if platform.sdk_root}
        if len(roots) != 1:
            return None
        base = roots.pop()
    candidate = Path(base) / "lib" / "llvm" / "bin" / "clang++"
    return candidate if candidate.is_file() else None
'''+s[end:]
s=s.replace("clang = compiler()","clang = compiler(engine)")
f.write_text(s)
for rel,old,new in (
    ("tools/bigcherry/engine_adapters/radiance_bump.py","compose_check.compiler()","compose_check.compiler(engine)"),
    ("tools/lab/radiance/unit_harness.py","compose_check.compiler()",'compose_check.compiler("radiance")'),
    ("tools/tests/patch/test_compose_check.py","cc.compiler()","cc.compiler(ENGINE)"),
):
    f=p/rel
    s=f.read_text()
    assert s.count(old)==1,rel
    f.write_text(s.replace(old,new))
