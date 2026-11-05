"""BRVP06: the ROCm SDK the CI build image carries must be the SDK the CI lane's stack names (card-free check)."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path


class SdkIdentityError(RuntimeError):
    pass


CI_PLATFORM = "linux-radiance"
CI_HOST = "dev"


def ci_sdk_identity(root: Path) -> str:
    """The ROCm SDK the CI build image carries, tied to the stack the CI lane names.

    The lane's platform names a stack, the stack names an SDK version, the tracked example host maps that version to
    the install path the image has, and the Dockerfile ARG installs it. All four must agree or the key cannot be made:
    a build image on one SDK must never be keyed as another."""
    platform = tomllib.loads((root / "engines/radiance/recipes.toml").read_text())["platform"][CI_PLATFORM]
    stack = platform.get("stack")
    stacks = tomllib.loads((root / "engines/radiance/recipes.toml").read_text()).get("stack", {})
    if stack not in stacks or not stacks[stack].get("sdk"):
        raise SdkIdentityError(f"platform.{CI_PLATFORM}.stack={stack!r} does not name a stack with a ROCm SDK")
    sdk = stacks[stack]["sdk"]
    sdks = tomllib.loads((root / "config/environment.example.toml").read_text())["host"][CI_HOST]["rocm-sdks"]
    if sdk not in sdks:
        raise SdkIdentityError(f"the example host declares no ROCm SDK {sdk!r} (stack {stack})")
    match = re.search(r"^ARG ROCM_VERSION=(\S+)$", (root / ".github/docker/radiance-build/Dockerfile").read_text(), re.M)
    if match is None:
        raise SdkIdentityError("the build image Dockerfile has no ARG ROCM_VERSION")
    version = match.group(1)
    if not (version == sdk or version.startswith(sdk + ".")):
        raise SdkIdentityError(f"the build image installs ROCm {version}; stack {stack} names SDK {sdk}")
    if Path(sdks[sdk]).name != f"rocm-{version}":
        raise SdkIdentityError(f"the example host maps SDK {sdk} to {sdks[sdk]}; the image installs /opt/rocm-{version}")
    return f"stack={stack};sdk={sdk};version={version};root={sdks[sdk]}"

