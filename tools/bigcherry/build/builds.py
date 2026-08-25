"""Content-addressed build plans, effective build IDs, and reuse checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..context import ProjectContext


class BuildIdentityError(ValueError):
    pass


def _canonical(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(item) for item in value]
    return value


def _digest(domain: str, value: object) -> str:
    encoded = json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.blake2b(
        domain.encode() + b"\0" + encoded, digest_size=16
    ).hexdigest()


@dataclass(frozen=True)
class BuildPlan:
    """RV48/RE07 audit fix: ``input_hashes`` replaces the old hard-coded
    ``inventory_hash``/``winners_hash`` pair. ``config.Build.needs`` is
    already the sole authority for what a lane requires (RE17) -- a build
    plan's OWN identity must track that generically too, or a third kind
    of declared need (e.g. HI66's correctness-evidence artifact) has
    nowhere to participate in ``build_plan_id`` without a fourth named
    field and a fourth special case in every reader. ``inventory_hash``/
    ``winners_hash`` remain as read-only compatibility properties.

    ``catalog_architectures`` is the architecture set actually requested
    for candidate GENERATION -- distinct from ``targets`` (the compiled
    AMDGPU targets). Two lanes sharing every other BuildPlan field but
    requesting generation for different architectures previously produced
    the SAME build_plan_id/build_dir despite representing two different
    generated catalogs -- campaign_workers.py's own generated_compile_
    inputs_hash reuse-miss check existed specifically to paper over this
    gap post hoc; this field closes it at the identity layer instead.
    Empty for a build with no generate stage (stock).

    ``backend`` (RE-backend-identity, external review 2026-08-20): explicit
    identity for which backend adapter produced ``cmake_options`` --
    defaults to ``"hip"`` so this field's addition alone does not change
    any existing HIP BuildPlan's identity (only threading the real
    backend-injected options into ``cmake_options``, done at the same time
    in the caller, changes HIP identity -- see campaign_lane.py). Without
    this field, two BuildPlans differing only in which backend adapter
    computed their (possibly textually-similar) cmake_options could not be
    distinguished by inspection alone, even though build_plan_id already
    differs once cmake_options genuinely differs.

    ``requested_targets`` (HI110, dev-gpt-agent design review 2026-08-24):
    the sorted set of CMake target names (binary_relative_path's own name
    plus every extra_binary_names entry) this build actually asked ninja to
    produce. Two lanes sharing every other BuildPlan field but requesting
    different target sets (e.g. bin/llama-bench vs bin/llama-server, same
    source/config/toolchain) previously produced the SAME build_plan_id and
    were pointed at the SAME build_dir -- campaign_workers.py's own comment
    called this "a shared configure cache across targets," but in practice
    a second target request mutates shared libraries already published
    under the first target's build_plan_id (ninja relinks libggml-hip.so
    when asked for a different top-level target), corrupting an artifact
    the store treats as immutable and poisoning reuse-validation for BOTH
    targets. Including the target set in identity gives each distinct
    target set its own build_plan_id/build_dir, matching
    ``catalog_architectures``'s precedent for closing this exact class of
    identity gap.
    """

    source_slice_id: str
    phase: str
    platform: str
    targets: tuple[str, ...]
    cmake_options: tuple[tuple[str, str], ...] = ()
    backend: str = "hip"
    variant_set: str | None = None
    catalog_architectures: tuple[str, ...] = ()
    requested_targets: tuple[str, ...] = ()
    #: (need_name, content_hash) pairs, sorted by need_name -- one entry
    #: per key in config.Build.needs this lane actually resolved, generic
    #: over whatever kinds exist now or are added later.
    input_hashes: tuple[tuple[str, str], ...] = ()
    resource_report_hashes: tuple[str, ...] = ()
    toolchain_request: tuple[tuple[str, str], ...] = ()
    environment: tuple[tuple[str, str], ...] = ()

    def canonical(self) -> dict[str, object]:
        return _canonical(asdict(self))  # type: ignore[return-value]

    @property
    def build_plan_id(self) -> str:
        return _digest("bigcherry/build-plan/v1", self.canonical())

    @property
    def inventory_hash(self) -> str | None:
        return dict(self.input_hashes).get("inventory")

    @property
    def winners_hash(self) -> str | None:
        return dict(self.input_hashes).get("promoted-winners")


def effective_build_id(configure_record: Mapping[str, object]) -> str:
    """Hash the normalized post-configure record, not the requested label.

    Mapping (not dict): covariant in the value type, so a
    ``dict[str, str]`` from parse_effective_configure() is accepted directly
    -- dict invariance would force every caller to widen the annotation."""
    if not isinstance(configure_record, Mapping) or not configure_record:
        raise BuildIdentityError(
            "effective configure record must be a non-empty object"
        )
    return _digest("bigcherry/effective-build/v1", dict(configure_record))


def binary_hash(binary: Path) -> str:
    try:
        digest = hashlib.sha256()
        with binary.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise BuildIdentityError(f"cannot hash binary {binary}: {exc}") from exc


def build_directory(
    context: ProjectContext, source_slice_id: str, plan: BuildPlan
) -> Path:
    return context.work_root / "builds" / source_slice_id / plan.build_plan_id


def resolve_runtime_artifacts(
    binary: Path, *, extra_binaries: tuple[Path, ...] = ()
) -> tuple[Path, ...]:
    """Every regular (non-symlink) file this build's own compile step
    produced alongside ``binary`` -- llama.cpp's CMake build places every
    project shared library directly next to its executables (libggml*.so*,
    libllama*.so*), never installed elsewhere. This is a directory-
    membership proxy for "this build's runtime output closure", not a
    linker-dependency walk (ldd/readelf) -- deliberately: system libraries
    (libc, the ROCm runtime itself) live outside this directory and are NOT
    part of what this specific build produced; a change there is a
    toolchain/environment identity question, not a build-output one.

    gpt-auto-agent review finding: a reuse check that only hashes the
    requested launcher (e.g. llama-bench) can accept a cache hit even if
    the actual HIP dispatch implementation (libggml-hip.so) changed --
    RE09's own investigation established that the meaningful logic lives
    there, not in the small launcher stub.

    ``extra_binaries`` (RE26): a lane can request additional compiled
    executables from the SAME build_dir/configure (e.g. test-backend-ops
    alongside the tune lane's llama-bench) that need to travel in the same
    runtime bundle as ``binary`` -- they share its shared-library closure,
    so they belong in the same content-addressed bundle rather than a
    second, redundant one. Callers are responsible for having actually
    requested these as cmake build targets; this function only resolves
    and includes them, it does not build them.
    """
    directory = binary.parent
    artifacts: list[Path] = [binary]
    artifacts.extend(extra_binaries)

    seen = {path for path in artifacts}
    # Linux uses *.so*; Windows builds (this project's HIP-on-Windows path)
    # use *.dll -- the pre-Windows version of this function only globbed
    # *.so*, so a stale ggml-hip.dll with an unchanged launcher would not
    # have invalidated reuse on Windows at all. *.dylib included for
    # completeness though there is no macOS HIP path today.
    for pattern in ("*.so*", "*.dll", "*.dylib"):
        for candidate in sorted(directory.glob(pattern)):
            if candidate.is_file() and not candidate.is_symlink() and candidate not in seen:
                artifacts.append(candidate)
                seen.add(candidate)

    return tuple(artifacts)


def runtime_bundle_hash(artifacts: dict[str, str]) -> str:
    """Canonical hash of a ``{filename: sha256}`` runtime artifact map --
    what validate_reuse() now trusts instead of a single binary_hash alone.
    """
    return _digest("bigcherry/runtime-bundle/v1", artifacts)


#: CMakeCache.txt keys that actually describe what got built: resolved
#: compiler paths, build type, GPU targets, and every autotune/HIP option.
#: Deliberately not the whole cache -- that also carries unrelated
#: find_package probe results and generator-internal bookkeeping (line
#: numbers, help strings) that say nothing about build identity and would
#: make effective_build_id() sensitive to changes that do not matter.
_EFFECTIVE_CONFIGURE_PREFIXES = (
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_BUILD_TYPE",
    "AMDGPU_TARGETS",
    "GGML_HIP",
    # RE-backend-identity (external review, 2026-08-20): Vulkan's real
    # configure options were silently excluded from effective-build
    # identity -- a Vulkan-relevant CMake change would not change
    # effective_build_id, breaking the same reuse-safety guarantee the
    # HIP prefixes above exist for.
    "GGML_VULKAN",
    # HI82 (gpt-auto-agent, req_cc5af49494fe457a): once a caller opts into
    # reproducible-linking flags (e.g. Windows /Brepro to make lld-link
    # stop embedding a wall-clock PE timestamp), those flags must be part
    # of effective build identity -- otherwise a build with and without
    # them would silently collapse to the same effective_build_id despite
    # producing non-comparable (non-reproducible vs reproducible) binaries.
    "CMAKE_EXE_LINKER_FLAGS",
    "CMAKE_SHARED_LINKER_FLAGS",
    "CMAKE_MODULE_LINKER_FLAGS",
)


def parse_effective_configure(cmake_cache: Path) -> dict[str, str]:
    """The post-configure record ``validate_reuse``/``effective_build_id``
    need: read back from CMakeCache.txt, not from the requested BuildPlan.
    ``BuildPlan`` records what was asked for; this records what CMake
    actually resolved, which is what a reuse decision needs to trust.
    """
    if not cmake_cache.is_file():
        raise BuildIdentityError(
            f"no CMakeCache.txt at {cmake_cache} -- configure did not run"
        )
    record: dict[str, str] = {}
    for line in cmake_cache.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        key_type, sep, value = line.partition("=")
        if not sep:
            continue
        key = key_type.partition(":")[0]
        if key.startswith(_EFFECTIVE_CONFIGURE_PREFIXES):
            record[key] = value
    if not record:
        raise BuildIdentityError(f"no relevant configure keys found in {cmake_cache}")
    return record


# ------------------------------------------------------------------------
# HI82 (design/implementation: gpt-auto-agent, req_51838ef1ea5f4086 +
# req_37641bb91ff4442c + req_527bff46e32e481c): compiled-command
# verification.
#
# parse_effective_configure() above proves what CMake RESOLVED into
# CMakeCache.txt. It has never proven that resolved intent actually reached
# the compiler invocation -- found for real this session on Windows/Ninja+
# Clang, where a configured CMAKE_HIP_FLAGS value silently never appeared on
# the real HIP compile command line (HI81). post_build_verify() closes that
# gap by reading compile_commands.json (falling back to `ninja -t commands`)
# and failing closed if the requested architecture or CMAKE_HIP_FLAGS is
# absent from the real HIP compile commands. compile_verification_id folds
# that evidence into the SAME identity/reuse contract validate_reuse()
# already enforces, rather than inventing a second, parallel build-identity
# authority (the first draft of this work did exactly that, as a standalone
# module -- reverted after review found the duplication).
# ------------------------------------------------------------------------

CMakeArgs = Sequence[str] | Mapping[str, object]


@dataclass(frozen=True)
class CommandRequirement:
    """Additional verification over real compiler command lines.

    ``selector_regex`` is evaluated against the canonical subject:

        <source file>
        <compiler command>

    Required tokens may be required either somewhere in the selected
    command set or in every selected command. Forbidden tokens must occur
    nowhere in the selected set.
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompletedBuildEvidence:
    """One completed build's existing BigCherry identities plus HI82 proof.

    This deliberately has NO second generic build-identity digest.
    ``effective_build_id`` remains the effective configuration identity;
    ``runtime_bundle_hash`` remains the output identity; and
    ``compile_verification_id`` names the new proof that configured intent
    reached the compiler.
    """

    effective_configure: dict[str, str]
    effective_build_id: str
    runtime_artifacts: dict[str, str]
    runtime_bundle_hash: str
    verification: BuildVerificationEvidence
    compile_verification_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_configure": dict(self.effective_configure),
            "effective_build_id": self.effective_build_id,
            "runtime_artifacts": dict(self.runtime_artifacts),
            "runtime_bundle_hash": self.runtime_bundle_hash,
            "verification": self.verification.to_dict(),
            "compile_verification_id": self.compile_verification_id,
        }

    def campaign_identity(self) -> dict[str, object]:
        """Stable subset suitable for binding campaign resume state."""
        return {
            "effective_build_id": self.effective_build_id,
            "compile_verification_id": self.compile_verification_id,
            "compile_commands_digest": self.verification.compile_commands_digest,
            "hip_compile_commands_digest": self.verification.hip_compile_commands_digest,
            "runtime_bundle_hash": self.runtime_bundle_hash,
            "runtime_artifacts": dict(sorted(self.runtime_artifacts.items())),
        }


@dataclass(frozen=True)
class _Command:
    source: str
    directory: str
    text: str


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
        key = key.split(":", 1)[0]
        result[key] = value if sep else "ON"
    return result


def _load_commands(build_dir: Path, env: Mapping[str, str]) -> tuple[str, tuple[_Command, ...]]:
    db = build_dir / "compile_commands.json"

    if db.is_file():
        try:
            payload = json.loads(db.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise BuildIdentityError(f"cannot read compile database {db}: {exc}") from exc

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
    """True only for actual HIP compilation, not linker commands.

    A Ninja fallback command line containing e.g. ``foo.cu.o`` on a LINK
    step is not itself a compile invocation; treating it as one would
    incorrectly require CMAKE_HIP_FLAGS on link lines too.
    """
    source = command.source.lower().replace("\\", "/")
    if source.endswith((".cu", ".hip")):
        return True

    text = " ".join(command.text.lower().replace("\\", "/").split())

    # Ninja's fallback also emits link commands. Device compilation must be
    # a compile invocation (has a "-c" flag), not a link.
    if " -c " not in f" {text} ":
        return False

    return (
        bool(re.search(r"\.(?:cu|hip)(?:[\"']|\s|$)", text))
        or "-x hip" in text
        or "__hip_platform_amd__" in text
        or "hipcc" in text
    )


def _commands_digest(commands: Sequence[_Command], source_root: Path, build_dir: Path) -> str:
    canonical = sorted(_subject(command, source_root, build_dir) for command in commands)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flag_tokens(value: str) -> tuple[str, ...]:
    try:
        return tuple(shlex.split(value, posix=True))
    except ValueError as exc:
        raise BuildIdentityError(f"cannot parse CMAKE_HIP_FLAGS={value!r}: {exc}") from exc


def _architecture_tokens(architecture: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(architecture, str):
        values = re.split(r"[;,\s]+", architecture)
    else:
        values = []
        for item in architecture:
            values.extend(re.split(r"[;,\s]+", str(item)))
    return tuple(value for value in values if value)


_REQUESTED_CACHE_KEYS = frozenset({
    "CMAKE_BUILD_TYPE", "CMAKE_HIP_FLAGS", "CMAKE_HIP_ARCHITECTURES",
    "CMAKE_EXPORT_COMPILE_COMMANDS", "AMDGPU_TARGETS",
    "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_PREFIX_PATH",
    "CMAKE_EXE_LINKER_FLAGS", "CMAKE_SHARED_LINKER_FLAGS", "CMAKE_MODULE_LINKER_FLAGS",
})


def _verify_requested_cache_values(
    defines: Mapping[str, str], cache: Mapping[str, str],
) -> VerificationCheck | None:
    """Reject a stale CMake cache that disagrees with requested build intent."""
    relevant = {
        key: value for key, value in defines.items()
        if key in _REQUESTED_CACHE_KEYS or key.startswith("GGML_") or key.startswith("BIGCHERRY_")
    }
    if not relevant:
        return None

    mismatches: list[str] = []
    for key, requested in sorted(relevant.items()):
        actual = cache.get(key)
        if actual is None:
            mismatches.append(f"{key}: requested {requested!r}, absent from CMake cache")
            continue

        requested_value = requested.strip().strip('"').replace("\\", "/")
        actual_value = actual.strip().strip('"').replace("\\", "/")

        if key in {"AMDGPU_TARGETS", "CMAKE_HIP_ARCHITECTURES"}:
            requested_parts = tuple(part for part in re.split(r"[;,]+", requested_value) if part)
            actual_parts = tuple(part for part in re.split(r"[;,]+", actual_value) if part)
            equal = requested_parts == actual_parts
        elif requested_value.upper() in {"ON", "OFF"}:
            equal = requested_value.upper() == actual_value.upper()
        else:
            equal = requested_value == actual_value

        if not equal:
            mismatches.append(f"{key}: requested {requested!r}, cache has {actual!r}")

    if mismatches:
        raise BuildIdentityError(
            "requested CMake configuration does not match the resolved CMake cache: "
            + "; ".join(mismatches)
        )

    return VerificationCheck(
        label="requested-cmake-config", status="pass", selected_commands=0,
        detail=f"{len(relevant)} requested CMake values match the cache",
    )


def _verify_requirement(
    requirement: CommandRequirement, commands: Sequence[_Command],
    source_root: Path, build_dir: Path,
) -> VerificationCheck:
    try:
        selector = re.compile(requirement.selector_regex)
    except re.error as exc:
        raise BuildIdentityError(f"invalid selector regex for {requirement.label!r}: {exc}") from exc

    subjects = [
        subject for command in commands
        if selector.search(subject := _subject(command, source_root, build_dir))
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
    build_dir: Path, *, source_root: Path, architecture: str | Sequence[str],
    requested_cmake_args: CMakeArgs = (), build_env: Mapping[str, str] | None = None,
    command_requirements: Sequence[CommandRequirement] = (),
) -> BuildVerificationEvidence:
    """Verify requested/resolved build intent against real HIP compilation.

    This is deliberately fail-closed. CMakeCache.txt proves what CMake
    resolved; compile_commands.json or ``ninja -t commands`` proves what the
    compiler was actually invoked with.
    """
    build_dir = Path(build_dir).resolve()
    source_root = Path(source_root).resolve()
    env = dict(os.environ if build_env is None else build_env)

    cmake_args = _normalise_cmake_args(requested_cmake_args)
    defines = _cmake_defines(cmake_args)
    cache = _read_cmake_cache_raw(build_dir)

    checks: list[VerificationCheck] = []

    requested_check = _verify_requested_cache_values(defines, cache)
    if requested_check is not None:
        checks.append(requested_check)

    command_source, commands = _load_commands(build_dir, env)

    hip_commands = tuple(command for command in commands if _is_hip_compile(command))
    if not hip_commands:
        raise BuildIdentityError(
            "post-build verification found no HIP compile commands; refusing HIP build evidence"
        )

    hip_subjects = [_subject(command, source_root, build_dir) for command in hip_commands]

    architectures = _architecture_tokens(architecture)
    for target in architectures:
        if not any(target in subject for subject in hip_subjects):
            raise BuildIdentityError(
                f"requested architecture {target!r} is absent from HIP compile commands"
            )

    checks.append(VerificationCheck(
        label="hip-architecture", status="pass", selected_commands=len(hip_commands),
        detail=f"architectures observed in HIP compile commands: {list(architectures)!r}",
    ))

    # Explicit requested value wins; otherwise the resolved cache is the
    # build's authoritative CMake intent. Do NOT use the current parent
    # process's CMAKE_HIP_FLAGS here -- that environment may have changed
    # since this build tree was configured.
    hip_flags = defines.get("CMAKE_HIP_FLAGS")
    if hip_flags is None:
        hip_flags = cache.get("CMAKE_HIP_FLAGS", "")

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


def build_verification_id(evidence: BuildVerificationEvidence | Mapping[str, object]) -> str:
    payload: object = evidence.to_dict() if isinstance(evidence, BuildVerificationEvidence) else dict(evidence)
    return _digest("bigcherry/build-verification/v1", payload)


def capture_completed_build_evidence(
    build_dir: Path, *, source_root: Path, architecture: str | Sequence[str], binary: Path,
    extra_binaries: tuple[Path, ...] = (), requested_cmake_args: CMakeArgs = (),
    build_env: Mapping[str, str] | None = None,
    command_requirements: Sequence[CommandRequirement] = (),
) -> CompletedBuildEvidence:
    """Capture the existing BigCherry identities plus compile verification.

    Unlike HI82's original standalone ``capture_build_identity()``, this
    does not invent another all-encompassing identity or write another
    manifest -- it reuses ``effective_build_id``/``runtime_bundle_hash``.
    """
    build_dir = Path(build_dir).resolve()
    binary = Path(binary)

    if not binary.is_file():
        raise BuildIdentityError(f"requested binary is missing: {binary}")
    for extra in extra_binaries:
        if not extra.is_file():
            raise BuildIdentityError(f"requested runtime artifact is missing: {extra}")

    effective_configure = parse_effective_configure(build_dir / "CMakeCache.txt")

    verification = post_build_verify(
        build_dir, source_root=source_root, architecture=architecture,
        requested_cmake_args=requested_cmake_args, build_env=build_env,
        command_requirements=command_requirements,
    )

    runtime_artifacts = {
        artifact.name: binary_hash(artifact)
        for artifact in resolve_runtime_artifacts(binary, extra_binaries=extra_binaries)
    }

    return CompletedBuildEvidence(
        effective_configure=effective_configure,
        effective_build_id=effective_build_id(effective_configure),
        runtime_artifacts=runtime_artifacts,
        runtime_bundle_hash=runtime_bundle_hash(runtime_artifacts),
        verification=verification,
        compile_verification_id=build_verification_id(verification),
    )


def _read_cmake_cache_raw(build_dir: Path) -> dict[str, str]:
    """Full (unfiltered-by-prefix) CMakeCache.txt keys, for post_build_verify's
    own requested-vs-resolved comparison -- distinct from
    parse_effective_configure()'s identity-relevant-prefix filter."""
    path = build_dir / "CMakeCache.txt"
    if not path.is_file():
        raise BuildIdentityError(f"no CMakeCache.txt at {path} -- configure did not run")

    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        lhs, value = line.split("=", 1)
        key = lhs.split(":", 1)[0]
        values[key] = value
    return values


def validate_reuse(
    metadata: dict[str, object],
    plan: BuildPlan,
    *,
    binary: Path,
    expected_toolchain: object | None = None,
    runtime_bundle_hash: str | None = None,
    compile_verification: BuildVerificationEvidence | None = None,
) -> None:
    if metadata.get("source_slice_id") != plan.source_slice_id:
        raise BuildIdentityError("source_slice_id does not match build plan")
    if metadata.get("build_plan_id") != plan.build_plan_id:
        raise BuildIdentityError("build_plan_id does not match requested plan")
    record = metadata.get("effective_configure")
    if not isinstance(record, dict):
        raise BuildIdentityError("effective configure metadata is missing")
    recorded_id = metadata.get("build_id")
    if recorded_id != effective_build_id(record):
        raise BuildIdentityError("recorded build_id does not recompute")
    if (
        expected_toolchain is not None
        and metadata.get("toolchain") != expected_toolchain
    ):
        raise BuildIdentityError("toolchain identity does not match")
    if not binary.is_file():
        raise BuildIdentityError("requested binary is missing")
    if metadata.get("binary_hash") != binary_hash(binary):
        raise BuildIdentityError("binary hash does not match recorded identity")
    # binary_hash alone only proves the requested launcher is unchanged --
    # RE09 established the meaningful HIP dispatch logic lives in
    # libggml-hip.so, not the launcher. When the caller supplies a
    # freshly-computed runtime_bundle_hash (see resolve_runtime_artifacts),
    # require it to match too, so a changed dependent library invalidates
    # reuse even when the launcher itself is byte-identical.
    if (
        runtime_bundle_hash is not None
        and metadata.get("runtime_bundle_hash") != runtime_bundle_hash
    ):
        raise BuildIdentityError("runtime bundle hash does not match recorded identity")

    # HI82: real compile-command evidence, folded into the same reuse
    # contract rather than a second parallel identity. Only checked when
    # the caller supplies fresh compile_verification (HIP builds today --
    # see capture_completed_build_evidence()); a caller that never captured
    # it (e.g. a non-HIP backend) is unaffected.
    if compile_verification is not None:
        recorded_evidence = metadata.get("compile_verification")
        recorded_verification_id = metadata.get("compile_verification_id")

        if not isinstance(recorded_evidence, dict):
            raise BuildIdentityError("compile verification metadata is missing")
        if not isinstance(recorded_verification_id, str):
            raise BuildIdentityError("compile_verification_id is missing")

        if build_verification_id(recorded_evidence) != recorded_verification_id:
            raise BuildIdentityError("recorded compile_verification_id does not recompute")

        current_verification_id = build_verification_id(compile_verification)
        if current_verification_id != recorded_verification_id:
            raise BuildIdentityError(
                "real compile-command verification does not match recorded build identity"
            )
