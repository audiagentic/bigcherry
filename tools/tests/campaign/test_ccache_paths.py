"""Stable compile paths for ccache (CI time): slots are exclusive, the launcher rewrites only this build's two directories,
concurrent builds with different sources never see each other's headers, and with a slot an unchanged unit hits."""
from __future__ import annotations

import multiprocessing
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))

from bigcherry.campaign import ccache_paths  # noqa: E402
from bigcherry.campaign import build as campaign_build  # noqa: E402

LAUNCHER = REPO / "tools" / "bigcherry" / "bin" / "bc-ccache"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("g++")


def _fake_ccache(directory: Path, log: Path | None = None) -> Path:
    """A `ccache` that only runs the compiler (and records its argument text): the launcher's behaviour does not depend on
    ccache's own caching, which is tested separately with the real one."""
    bindir = directory / "bin"
    bindir.mkdir(exist_ok=True)
    script = bindir / "ccache"
    record = f'printf "%s\\n" "$*" >> "{log}"\n' if log else ""
    script.write_text(f"#!/usr/bin/env bash\n{record}exec \"$@\"\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return bindir


def _build_one(args: tuple[str, str, str, str, str, int]) -> tuple[int | None, str, str]:
    """One 'build' in its own process: take a slot, compile its own tree through the launcher, report which value the object
    holds. The header is found through -I<source>/inc, which the launcher rewrites to the slot's link."""
    work, source, build, value, bindir, rounds = args
    from bigcherry.campaign import ccache_paths as paths_module
    seen = []
    slot = None
    with paths_module.stable_paths(Path(work), Path(source), Path(build)) as slot:
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        for round_number in range(rounds):
            out = Path(build) / f"unit{round_number}.o"
            subprocess.run([str(LAUNCHER), CC, "-c", f"-I{source}/inc", f"{source}/unit.c", "-o", str(out)],
                           env=env, check=True, cwd=build)
            symbols = subprocess.run(["nm", str(out)], capture_output=True, text=True, check=True).stdout
            seen.append(symbols)
            time.sleep(0.15)   # overlap with the other builds: every one of them holds its slot (or none) for the whole run
    return slot, value, "\n".join(seen)


class StableSlotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.saved = {name: os.environ.get(name) for name in ccache_paths.VARIABLES}
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _tree(self, name: str) -> tuple[Path, Path]:
        source, build = self.root / "sources" / name, self.root / "builds" / name / "plan"
        source.mkdir(parents=True)
        build.mkdir(parents=True)
        return source, build

    def test_slots_are_exclusive_and_released(self) -> None:
        (s1, b1), (s2, b2), (s3, b3) = self._tree("a" * 32), self._tree("b" * 32), self._tree("c" * 32)
        with ccache_paths.stable_paths(self.root, s1, b1, slots=2) as first:
            self.assertEqual(first, 0)
            self.assertEqual(os.environ["BC_CCACHE_REAL_SRC"], str(s1.resolve()))
            link = Path(os.environ["BC_CCACHE_LINK_SRC"])
            self.assertEqual(link.resolve(), s1.resolve())
            with ccache_paths.stable_paths(self.root, s2, b2, slots=2) as second:
                self.assertEqual(second, 1)          # a second holder gets its own slot, never the first's
                self.assertEqual(Path(os.environ["BC_CCACHE_LINK_SRC"]).resolve(), s2.resolve())
                self.assertEqual(link.resolve(), s1.resolve())   # the first slot still names the first build
                with ccache_paths.stable_paths(self.root, s3, b3, slots=2) as third:
                    self.assertIsNone(third)         # all taken: plain ccache, no variables
                    self.assertNotIn("BC_CCACHE_REAL_SRC", os.environ)
                self.assertEqual(os.environ["BC_CCACHE_REAL_SRC"], str(s2.resolve()))   # restored for the enclosing build
            self.assertEqual(os.environ["BC_CCACHE_REAL_SRC"], str(s1.resolve()))
        self.assertNotIn("BC_CCACHE_REAL_SRC", os.environ)
        with ccache_paths.stable_paths(self.root, s3, b3, slots=2) as again:
            self.assertEqual(again, 0)               # released on exit: the slot is free again

    def test_the_launcher_rewrites_exactly_this_builds_two_directories(self) -> None:
        log = self.root / "args.log"
        bindir = _fake_ccache(self.root, log)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
        subprocess.run([str(LAUNCHER), "true", "-I/r/sources/AAA/core", "/r/sources/AAA/x.c", "-I/r/builds/H/P/rad_include",
                        "-I/elsewhere/core", "-DKEEP=/r/sources/BBB"],
                       env={**env, "BC_CCACHE_REAL_SRC": "/r/sources/AAA", "BC_CCACHE_REAL_BLD": "/r/builds/H/P",
                            "BC_CCACHE_LINK_SRC": "/r/.cc-stable/slot0/src", "BC_CCACHE_LINK_BLD": "/r/.cc-stable/slot0/bld"},
                       check=True)
        self.assertEqual(log.read_text().strip(),
                         "true -I/r/.cc-stable/slot0/src/core /r/.cc-stable/slot0/src/x.c -I/r/.cc-stable/slot0/bld/rad_include "
                         "-I/elsewhere/core -DKEEP=/r/sources/BBB")
        log.unlink()
        subprocess.run([str(LAUNCHER), "true", "-I/r/sources/AAA/core"], env=env, check=True)   # no slot: unchanged
        self.assertEqual(log.read_text().strip(), "true -I/r/sources/AAA/core")

    @unittest.skipUnless(CC and shutil.which("nm"), "needs a C compiler and nm")
    def test_concurrent_builds_with_different_sources_never_compile_each_others_headers(self) -> None:
        """Eight 'builds' at once, four slots: each has its own header value. An object must carry its own build's value in
        every round, whichever slot (or no slot, plain ccache) the build got."""
        bindir = _fake_ccache(self.root)
        jobs = []
        for index in range(8):
            source, build = self._tree(f"{index:032x}")
            (source / "inc").mkdir()
            (source / "inc" / "value.h").write_text(f"#define TAG tag_{1000 + index}\n")
            (source / "unit.c").write_text('#include "value.h"\nint TAG = 1;\n')
            jobs.append((str(self.root), str(source), str(build), f"tag_{1000 + index}", str(bindir), 6))
        with multiprocessing.get_context("fork").Pool(8) as pool:
            results = pool.map(_build_one, jobs)
        slots = [slot for slot, _, _ in results]
        self.assertTrue(any(s is not None for s in slots) and any(s is None for s in slots),
                        f"eight overlapping builds on four slots: some got a slot and some ran plain: {slots}")
        for index, (slot, tag, symbols) in enumerate(results):
            # every round's object carries the symbol its own header defines, and no other build's
            self.assertEqual(symbols.count(tag), 6, (slot, tag, symbols))
            for other in range(8):
                if other != index:
                    self.assertNotIn(f"tag_{1000 + other}", symbols, (slot, tag))

    @unittest.skipUnless(CC and shutil.which("ccache"), "needs a C compiler and the real ccache")
    def test_with_a_slot_an_unchanged_unit_hits_in_a_build_with_another_id(self) -> None:
        cache = self.root / "ccache-dir"
        env = dict(os.environ, CCACHE_DIR=str(cache), CCACHE_NOHASHDIR="true", CCACHE_COMPILERCHECK="content")
        env.pop("CCACHE_DISABLE", None)

        def hits() -> int:
            out = subprocess.run(["ccache", "-s"], env=env, capture_output=True, text=True, check=True).stdout
            line = next(l for l in out.splitlines() if l.strip().startswith("Hits:"))
            return int(line.split()[1])

        def build(name: str, use_slot: bool) -> None:
            source, build_dir = self._tree(name)
            (source / "inc").mkdir()
            (source / "inc" / "value.h").write_text("#define VALUE 7\n")
            (source / "unit.c").write_text('#include "value.h"\nint f(void) { return VALUE; }\n')
            command = [str(LAUNCHER), CC, "-c", f"-I{source}/inc", f"{source}/unit.c", "-o", str(build_dir / "unit.o")]
            if use_slot:
                with ccache_paths.stable_paths(self.root, source, build_dir):
                    subprocess.run(command, env={**env, **{k: os.environ[k] for k in ccache_paths.VARIABLES}}, check=True, cwd=build_dir)
            else:
                subprocess.run(command, env=env, check=True, cwd=build_dir)

        build("a" * 32, False)
        build("b" * 32, False)
        self.assertEqual(hits(), 0, "without a slot the id is in the command and nothing hits (the defect this fixes)")
        build("c" * 32, True)
        build("d" * 32, True)
        self.assertEqual(hits(), 1, "with a slot the second build's unchanged unit hits")


class LauncherTokenTests(unittest.TestCase):
    def test_the_token_resolves_to_the_script_and_other_values_are_unchanged(self) -> None:
        resolved = campaign_build.resolve_launcher(campaign_build.CCACHE_LAUNCHER_TOKEN)
        self.assertEqual(Path(resolved), LAUNCHER)
        self.assertTrue(os.access(resolved, os.X_OK))
        self.assertEqual(campaign_build.resolve_launcher("ccache"), "ccache")

    def test_the_launcher_is_executable_in_git(self) -> None:
        # A checkout (CI) takes the mode from git, not from this working tree.
        out = subprocess.run(["git", "ls-files", "-s", str(LAUNCHER)], cwd=REPO, capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout:
            self.skipTest("not a git checkout")
        self.assertTrue(out.stdout.startswith("100755"), out.stdout)

    def test_the_radiance_recipe_names_the_token_for_every_compiler_launcher(self) -> None:
        text = (REPO / "engines/radiance/recipes.toml").read_text(encoding="utf-8")
        for language in ("C", "CXX", "HIP"):
            self.assertIn(f'CMAKE_{language}_COMPILER_LAUNCHER = "bc-ccache"', text)


if __name__ == "__main__":
    unittest.main()
