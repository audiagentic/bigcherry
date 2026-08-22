"""HI82 design item 7: build-configuration identity + post-build verification.

CMakeCache.txt proves what CMake believed it configured -- it is not proof
that a flag actually reached the compiler. HI81 (Windows CMAKE_HIP_FLAGS
does not propagate to the real HIP compile command line) is exactly this
gap, found this session by ad-hoc manual `ninja -t commands` inspection.
This module generalizes that check into something every campaign build
runs automatically and fails closed on: post_build_verify() reads
compile_commands.json where available, falls back to `ninja -t commands`,
and refuses to certify a build whose configured intent (CMAKE_HIP_FLAGS,
target architecture, any patch-specific CommandRequirement) is absent from
the commands that actually compiled.

capture_build_identity() wraps that verification with a content-addressed
identity digest (source tree + arch + build mode + generator + selected
environment + resolved CMake cache + resolved/hashed toolchain executables
+ canonicalized compile-command digests) and persists it next to the build
tree. No manifest is written when verification fails.

Design + implementation: GPT (gpt-auto-agent, request req_37641bb91ff4442c),
applied per plan item HI82 (design item 7).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
MANIFEST_NAME = "bigcherry-build-identity.json"

# These are deliberately configuration/flag inputs rather than the entire
# environment. PATH itself is represented indirectly by the resolved tool
# identities below; including the entire PATH would make build identity noisy
# without adding useful semantic information.
_ENV_KEYS = (
    "CC",
    "CXX",
    "HIPCXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CMAKE_ARGS",
    "CMAKE_GENERATOR",
    "CMAKE_BUILD_TYPE",
    "CMAKE_C_FLAGS",
    "CMAKE_CXX_FLAGS",
    "CMAKE_HIP_FLAGS",
    "HIPFLAGS",
    "HIPCC_COMPILE_FLAGS_APPEND",
    "AMDGPU_TARGETS",
    "GPU_TARGETS",
    "HIP_PATH",
    "ROCM_PATH",
)

_CACHE_KEYS = {
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_HIP_COMPILER",
    "CMAKE_HIP_COMPILER_ROCM_ROOT",
    "CMAKE_HIP_ARCHITECTURES",
    "CMAKE_C_FLAGS",
    "CMAKE_CXX_FLAGS",
    "CMAKE_HIP_FLAGS",
    "CMAKE_EXE_LINKER_FLAGS",
    "CMAKE_SHARED_LINKER_FLAGS",
}

_CACHE_PREFIXES = (
    "BIGCHERRY_",
    "GGML_",
    "LLAMA_",
    "HIP_",
    "ROCM_",
    "AMDGPU_",
    "GPU_",
)

CMakeArgs = Sequence[str] | Mapping[str, object]


class BuildIdentityError(RuntimeError):
    """The completed build cannot be proven to match its configured intent."""


@dataclass(frozen=True)
class ToolIdentity:
    requested: str | None
    resolved: str | None
    version: str | None
    content_sha256: str | None


@dataclass(frozen=True)
class CommandRequirement:
    """Additional build/patch-specific verification over real compiler commands.

    selector_regex is evaluated against:

        <source file>
        <canonicalised command>

    required_tokens are either required in at least one selected command or,
    when require_required_tokens_in_every_selected_command is true, in every
    selected command.

    forbidden_tokens must occur in none of the selected commands.
    """

    label: str
    selector_regex: str
    required_tokens: tuple[str, ...] = ()
    forbidden_tokens: tuple[str, ...] = ()
    require_required_tokens_in_every_selected_command: bool = False
    min_selected_commands: int = 1


@dataclass(frozen=True)
class VerificationCheck:
    label: str
    status: str
    selected_commands: int
    detail: str


@dataclass(frozen=True)
class BuildVerificationEvidence:
    command_source: str
    compile_command_count: int
    hip_compile_command_count: int
    compile_commands_digest: str
    hip_compile_commands_digest: str
    checks: tuple[VerificationCheck, ...]


@dataclass(frozen=True)
class BuildIdentity:
    schema_version: int
    identity_digest: str

    source_tree: str
    architecture: str
    build_mode: str
    generator: str

    requested_cmake_args: tuple[str, ...]
    environment: dict[str, str]
    cmake_cache: dict[str, str]

    tools: dict[str, ToolIdentity]
    rocm: dict[str, str]

    verification: BuildVerificationEvidence

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Command:
    source: str
    directory: str
    text: str


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    """Atomically replace path with JSON.

    NamedTemporaryFile(delete=False) / close / os.replace is intentional:
    replacing an open temporary file is not portable to Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True,
    )
    tmp = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _normalise_cmake_args(args: CMakeArgs) -> tuple[str, ...]:
    if not isinstance(args, Mapping):
        return tuple(str(arg) for arg in args)

    rendered: list[str] = []
    for key in sorted(args):
        value = args[key]
        if isinstance(value, bool):
            text = "ON" if value else "OFF"
        elif value is None:
            text = ""
        else:
            text = str(value)
        rendered.append(f"-D{key}={text}")
    return tuple(rendered)


def _cmake_defines(args: Sequence[str]) -> dict[str, str]:
    """Return effective -D values; later duplicate definitions win."""
    result: dict[str, str] = {}
    for arg in args:
        if not arg.startswith("-D"):
            continue
        item = arg[2:]
        key, sep, value = item.partition("=")
        # Accept -DNAME:TYPE=value as well as -DNAME=value.
        key = key.split(":", 1)[0]
        result[key] = value if sep else "ON"
    return result


def _read_cache(build_dir: Path) -> dict[str, str]:
    path = build_dir / "CMakeCache.txt"
    if not path.is_file():
        raise BuildIdentityError(f"missing CMake cache: {path}")

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        lhs, value = line.split("=", 1)
        key = lhs.split(":", 1)[0]
        if key in _CACHE_KEYS or key.startswith(_CACHE_PREFIXES):
            values[key] = value
    return dict(sorted(values.items()))


def _resolve_tool(name: str | None, env: Mapping[str, str]) -> Path | None:
    if not name:
        return None
    raw = name.strip().strip('"')
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    resolved = shutil.which(raw, path=env.get("PATH"))
    return Path(resolved).resolve() if resolved else None


def _tool_identity(
    requested: str | None, env: Mapping[str, str], memo: dict[str, ToolIdentity],
) -> ToolIdentity:
    path = _resolve_tool(requested, env)
    if path is None:
        return ToolIdentity(requested=requested, resolved=None, version=None, content_sha256=None)

    memo_key = os.path.normcase(str(path))
    if memo_key in memo:
        prior = memo[memo_key]
        return ToolIdentity(
            requested=requested, resolved=prior.resolved,
            version=prior.version, content_sha256=prior.content_sha256,
        )

    version: str | None = None
    try:
        result = subprocess.run(
            [str(path), "--version"], check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=dict(env), timeout=10,
        )
        version = (result.stdout + "\n" + result.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        content_sha256 = _sha256_file(path)
    except OSError:
        content_sha256 = None

    identity = ToolIdentity(
        requested=requested, resolved=str(path), version=version, content_sha256=content_sha256,
    )
    memo[memo_key] = identity
    return identity


def _tools(cache: Mapping[str, str], env: Mapping[str, str]) -> dict[str, ToolIdentity]:
    memo: dict[str, ToolIdentity] = {}
    requested = {
        "c_compiler": cache.get("CMAKE_C_COMPILER") or env.get("CC"),
        "cxx_compiler": cache.get("CMAKE_CXX_COMPILER") or env.get("CXX"),
        "hip_compiler": cache.get("CMAKE_HIP_COMPILER") or env.get("HIPCXX"),
        "cmake": "cmake",
        "ninja": "ninja",
        "hipcc": "hipcc",
        "hipconfig": "hipconfig",
    }
    return {name: _tool_identity(value, env, memo) for name, value in requested.items()}


def _load_commands(build_dir: Path, env: Mapping[str, str]) -> tuple[str, tuple[_Command, ...]]:
    """Read commands from compile_commands.json, falling back to Ninja."""
    db = build_dir / "compile_commands.json"

    if db.is_file():
        payload = json.loads(db.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise BuildIdentityError(f"invalid compile database: {db}")

        commands: list[_Command] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = item.get("command")
            if not text and isinstance(item.get("arguments"), list):
                text = shlex.join(str(arg) for arg in item["arguments"])
            if not text:
                continue
            commands.append(_Command(
                source=str(item.get("file", "")), directory=str(item.get("directory", "")),
                text=str(text),
            ))

        if not commands:
            raise BuildIdentityError(f"compile database contains no commands: {db}")
        return "compile_commands.json", tuple(commands)

    ninja = shutil.which("ninja", path=env.get("PATH"))
    if not ninja:
        raise BuildIdentityError(
            f"{db} is absent and ninja is unavailable for '-t commands' fallback"
        )

    result = subprocess.run(
        [ninja, "-C", str(build_dir), "-t", "commands"],
        check=False, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=dict(env), timeout=60,
    )
    if result.returncode != 0:
        raise BuildIdentityError(
            f"ninja -t commands failed ({result.returncode}): {result.stderr.strip()}"
        )

    commands = tuple(
        _Command(source="", directory=str(build_dir), text=line.strip())
        for line in result.stdout.splitlines() if line.strip()
    )
    if not commands:
        raise BuildIdentityError("ninja -t commands returned no commands")
    return "ninja -t commands", commands


def _replace_root(text: str, root: Path, token: str) -> str:
    """Make command digests independent of worktree/build directory names."""
    values = {str(root), str(root.resolve())}
    for value in list(values):
        values.add(value.replace("\\", "/"))

    result = text.replace("\\", "/")
    for value in sorted(values, key=len, reverse=True):
        result = result.replace(value.replace("\\", "/"), token)
    return result


def _subject(command: _Command, source_root: Path, build_dir: Path) -> str:
    source = _replace_root(command.source, source_root, "<SOURCE>")
    source = _replace_root(source, build_dir, "<BUILD>")

    text = _replace_root(command.text, source_root, "<SOURCE>")
    text = _replace_root(text, build_dir, "<BUILD>")

    return source + "\n" + " ".join(text.split())


def _is_hip_compile(command: _Command) -> bool:
    subject = (command.source + "\n" + command.text).lower().replace("\\", "/")
    # llama.cpp's HIP backend primarily compiles .cu sources. The other
    # markers cover explicit HIP language invocation and hipcc-based builds.
    return any(
        marker in subject
        for marker in (".cu", ".hip", "-x hip", "__hip_platform_amd__", "hipcc")
    )


def _commands_digest(commands: Sequence[_Command], source_root: Path, build_dir: Path) -> str:
    canonical = sorted(_subject(command, source_root, build_dir) for command in commands)
    return _sha256_bytes(_json_bytes(canonical))


def _flag_tokens(value: str) -> tuple[str, ...]:
    try:
        # HIP/clang flags use Unix-style quoting even when invoked from
        # Windows Ninja builds.
        return tuple(shlex.split(value, posix=True))
    except ValueError as exc:
        raise BuildIdentityError(f"cannot parse CMAKE_HIP_FLAGS={value!r}: {exc}") from exc


def _verify_requirement(
    requirement: CommandRequirement, commands: Sequence[_Command],
    source_root: Path, build_dir: Path,
) -> VerificationCheck:
    try:
        selector = re.compile(requirement.selector_regex)
    except re.error as exc:
        raise BuildIdentityError(
            f"invalid selector regex for {requirement.label!r}: {exc}"
        ) from exc

    subjects = [
        _subject(command, source_root, build_dir)
        for command in commands
        if selector.search(_subject(command, source_root, build_dir))
    ]

    if len(subjects) < requirement.min_selected_commands:
        raise BuildIdentityError(
            f"post-build check {requirement.label!r} selected {len(subjects)} commands; "
            f"need at least {requirement.min_selected_commands}"
        )

    missing: list[str] = []
    for token in requirement.required_tokens:
        if requirement.require_required_tokens_in_every_selected_command:
            present = all(token in subject for subject in subjects)
        else:
            present = any(token in subject for subject in subjects)
        if not present:
            missing.append(token)

    forbidden = [
        token for token in requirement.forbidden_tokens
        if any(token in subject for subject in subjects)
    ]

    if missing or forbidden:
        raise BuildIdentityError(
            f"post-build check {requirement.label!r} failed: "
            f"missing={missing!r}, forbidden_present={forbidden!r}"
        )

    return VerificationCheck(
        label=requirement.label, status="pass", selected_commands=len(subjects),
        detail="required/forbidden command-line tokens satisfied",
    )


def post_build_verify(
    build_dir: Path, *, source_root: Path, architecture: str,
    requested_cmake_args: CMakeArgs = (), build_env: Mapping[str, str] | None = None,
    command_requirements: Sequence[CommandRequirement] = (),
) -> BuildVerificationEvidence:
    """Verify configured intent against commands that actually compiled.

    This deliberately fails closed. CMakeCache.txt proves what CMake
    believed it configured. It is not sufficient evidence that the HIP
    compiler actually received those settings. This function therefore
    requires either compile_commands.json or a successful
    `ninja -t commands`.
    """
    build_dir = Path(build_dir).resolve()
    source_root = Path(source_root).resolve()
    env = dict(os.environ if build_env is None else build_env)

    cmake_args = _normalise_cmake_args(requested_cmake_args)
    defines = _cmake_defines(cmake_args)
    cache = _read_cache(build_dir)

    command_source, commands = _load_commands(build_dir, env)

    hip_commands = tuple(command for command in commands if _is_hip_compile(command))
    if not hip_commands:
        raise BuildIdentityError(
            "post-build verification found no HIP compile commands; refusing HIP build identity"
        )

    hip_subjects = [_subject(command, source_root, build_dir) for command in hip_commands]

    # The architecture must be visible in the real device compilation
    # command line, not merely in CMAKE_HIP_ARCHITECTURES.
    if architecture and not any(architecture in subject for subject in hip_subjects):
        raise BuildIdentityError(
            f"requested architecture {architecture!r} is absent from HIP compile commands"
        )

    checks: list[VerificationCheck] = [
        VerificationCheck(
            label="hip-architecture", status="pass", selected_commands=len(hip_commands),
            detail=f"architecture {architecture!r} observed in HIP compile commands",
        )
    ]

    # This is the important HI81-class guard.
    #
    # Prefer explicit invocation intent, then environment intent, then the
    # resolved cache value. A non-empty CMAKE_HIP_FLAGS must actually appear
    # on every HIP source compilation command.
    hip_flags = (
        defines.get("CMAKE_HIP_FLAGS") or env.get("CMAKE_HIP_FLAGS") or cache.get("CMAKE_HIP_FLAGS")
    )

    if hip_flags:
        tokens = _flag_tokens(hip_flags)
        missing: list[tuple[int, list[str]]] = []

        for index, subject in enumerate(hip_subjects):
            absent = [token for token in tokens if token not in subject]
            if absent:
                missing.append((index, absent))

        if missing:
            sample = ", ".join(f"#{index}: {tokens!r}" for index, tokens in missing[:5])
            more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            raise BuildIdentityError(
                "configured CMAKE_HIP_FLAGS did not propagate to all HIP compile commands: "
                f"{sample}{more}"
            )

        checks.append(VerificationCheck(
            label="cmake-hip-flags-propagation", status="pass", selected_commands=len(hip_commands),
            detail=f"all configured CMAKE_HIP_FLAGS tokens observed: {list(tokens)!r}",
        ))

    # Data-driven extension point for things that are not represented as
    # CMAKE_HIP_FLAGS. This replaces one-off manual `ninja -t commands`
    # inspection without putting patch-specific policy in this module.
    for requirement in command_requirements:
        checks.append(_verify_requirement(requirement, commands, source_root, build_dir))

    return BuildVerificationEvidence(
        command_source=command_source,
        compile_command_count=len(commands),
        hip_compile_command_count=len(hip_commands),
        compile_commands_digest=_commands_digest(commands, source_root, build_dir),
        hip_compile_commands_digest=_commands_digest(hip_commands, source_root, build_dir),
        checks=tuple(checks),
    )


def capture_build_identity(
    build_dir: Path, *, source_root: Path, source_tree: str, architecture: str, build_mode: str,
    requested_cmake_args: CMakeArgs = (), generator: str | None = None,
    build_env: Mapping[str, str] | None = None,
    command_requirements: Sequence[CommandRequirement] = (),
    manifest_path: Path | None = None,
) -> BuildIdentity:
    """Verify a completed build, digest it, and persist its identity.

    No manifest is written if post-build verification fails.
    """
    build_dir = Path(build_dir).resolve()
    source_root = Path(source_root).resolve()
    env = dict(os.environ if build_env is None else build_env)

    cmake_args = _normalise_cmake_args(requested_cmake_args)
    cache = _read_cache(build_dir)

    actual_generator = cache.get("CMAKE_GENERATOR", "").strip()
    if generator and actual_generator and generator != actual_generator:
        raise BuildIdentityError(
            f"configured generator {actual_generator!r} != requested {generator!r}"
        )
    actual_generator = actual_generator or generator or "unknown"

    verification = post_build_verify(
        build_dir, source_root=source_root, architecture=architecture,
        requested_cmake_args=cmake_args, build_env=env,
        command_requirements=command_requirements,
    )

    environment = {key: env[key] for key in _ENV_KEYS if env.get(key)}
    tools = _tools(cache, env)

    rocm = {
        key: value
        for key in ("CMAKE_HIP_COMPILER_ROCM_ROOT", "HIP_ROOT_DIR", "ROCM_PATH", "HIP_PATH")
        if (value := (cache.get(key) or env.get(key)))
    }

    # identity_digest intentionally excludes physical source/build roots.
    # source_tree identifies source content, while compile commands are
    # canonicalised to <SOURCE>/<BUILD> before their digests are produced.
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "source_tree": source_tree,
        "architecture": architecture,
        "build_mode": build_mode,
        "generator": actual_generator,
        "requested_cmake_args": list(cmake_args),
        "environment": environment,
        "cmake_cache": cache,
        "tools": {name: asdict(tool) for name, tool in tools.items()},
        "rocm": rocm,
        "verification": asdict(verification),
    }

    identity = BuildIdentity(
        schema_version=SCHEMA_VERSION,
        identity_digest=_sha256_bytes(_json_bytes(unsigned)),
        source_tree=source_tree, architecture=architecture, build_mode=build_mode,
        generator=actual_generator, requested_cmake_args=cmake_args,
        environment=environment, cmake_cache=cache,
        tools=tools, rocm=rocm, verification=verification,
    )

    output_path = manifest_path or build_dir / MANIFEST_NAME
    _atomic_write_json(output_path, identity.to_dict())
    return identity
