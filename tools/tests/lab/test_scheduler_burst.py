"""Card-free contract tests for the scheduler burst lab launcher."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "scheduler_burst", ROOT / "tools/lab/radiance/scheduler_burst.py")
burst = importlib.util.module_from_spec(spec)
spec.loader.exec_module(burst)


def native_build(tmp_path):
    build = tmp_path / "build"
    cwd = build / "tests"
    link_dir = cwd / "CMakeFiles/sched_test.dir"
    link_dir.mkdir(parents=True)
    source = tmp_path / "src/tests/sched_test.cpp"
    source.parent.mkdir(parents=True)
    source.touch()
    scheduler = source.parent.parent / "core/sched/scheduler.cpp"
    scheduler.parent.mkdir(parents=True)
    scheduler.touch()
    obj = "CMakeFiles/sched_test.dir/sched_test.cpp.o"
    (build / "CMakeCache.txt").write_text(
        "RAD_WITH_HIP:STRING=OFF\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
    (build / "compile_commands.json").write_text(json.dumps([{
        "directory": str(cwd), "file": str(source),
        "arguments": ["c++", "-O3", "-o", obj, "-c", str(source)],
    }]))
    (link_dir / "link.txt").write_text(
        f"c++ {obj} -o sched_test -Wl,--dependency-file=link.d ../core/librad_core.a")
    return build, source


def test_reuses_native_flags_and_archive_without_writing_to_build(tmp_path):
    build, source = native_build(tmp_path)
    scratch = tmp_path / "scratch"
    cwd, actual, comp, link, binary = burst.commands(build, scratch)
    assert cwd == build / "tests"
    assert actual == source
    assert str(burst.FIXTURE) in comp
    assert "-O3" in comp and comp[-1] == "-O0"
    assert "-I" + str(source.parent) in comp
    assert str(scratch / "burst.o") in comp and str(scratch / "burst.o") in link
    assert "../core/librad_core.a" in link
    assert str(binary) in link
    assert not any(x.startswith("-Wl,--dependency-file") for x in link)
    assert not scratch.exists()


def test_rejects_gpu_build(tmp_path):
    build, _ = native_build(tmp_path)
    (build / "CMakeCache.txt").write_text(
        "RAD_WITH_HIP:STRING=ON\nCMAKE_GENERATOR:INTERNAL=Unix Makefiles\n")
    with pytest.raises(ValueError, match="HIP-off"):
        burst.commands(build, tmp_path / "scratch")


def test_rejects_ambiguous_compile_entry(tmp_path):
    build, _ = native_build(tmp_path)
    path = build / "compile_commands.json"
    rows = json.loads(path.read_text())
    path.write_text(json.dumps(rows * 2))
    with pytest.raises(ValueError, match="one native scheduler"):
        burst.commands(build, tmp_path / "scratch")
