"""Named build definitions -- an upstream ref plus a patch selection."""
from __future__ import annotations
import hashlib, re, tomllib
from dataclasses import dataclass
from pathlib import Path
from . import paths

RECIPES_PATH = paths.REPO_ROOT / "recipes.toml"
FOLLOW_PIN = "pinned"
NEEDS = ("none", "inventory", "winners")

class RecipeError(ValueError): pass

@dataclass(frozen=True)
class Build:
    name: str
    description: str
    options: dict[str, str]
    variant_set: str | None = None
    needs: str = "none"

@dataclass(frozen=True)
class Platform:
    name: str
    description: str
    targets: str
    options: dict[str, str]
    c_compiler: str | None
    cxx_compiler: str | None

@dataclass(frozen=True)
class Recipe:
    name: str
    ref: str
    groups: frozenset[str] | None   # None = all groups; frozenset() = none
    states: frozenset[str] | None
    follows_pin: bool
    builds: tuple[str, ...] = ()
    platform: str | None = None
    default: bool = False

@dataclass(frozen=True)
class Config:
    pinned: str
    recipes: dict[str, Recipe]
    builds: dict[str, Build]
    platforms: dict[str, Platform]
    path: Path

    def recipe(self, name: str) -> Recipe:
        """Get a recipe by name, raising RecipeError if not found."""
        if name not in self.recipes:
            valid = sorted(self.recipes.keys())
            raise RecipeError(f"unknown recipe {name!r}; valid choices: {', '.join(valid)}")
        return self.recipes[name]

    def build(self, name: str) -> Build:
        """Get a build by name, raising RecipeError if not found."""
        if name not in self.builds:
            valid = sorted(self.builds.keys())
            raise RecipeError(f"unknown build {name!r}; valid choices: {', '.join(valid)}")
        return self.builds[name]

    def platform_for(self, recipe: Recipe) -> Platform:
        """Get the platform for a recipe, raising RecipeError if misconfigured."""
        if recipe.platform is None:
            raise RecipeError(f"recipe {recipe.name!r} does not name a platform")
        if recipe.platform not in self.platforms:
            valid = sorted(self.platforms.keys())
            raise RecipeError(f"recipe {recipe.name!r} names unknown platform {recipe.platform!r}; valid choices: {', '.join(valid)}")
        return self.platforms[recipe.platform]

def tree_state_key(ref: str, groups: frozenset[str] | None, states: frozenset[str] | None) -> str:
    """Fingerprint of the checkout state a recipe needs -- ref+groups+states
    ONLY (builds/platform/variant-set never trigger a reset, they are cmake
    args + generated output). Two recipes with the same key can build back to
    back with no reset between them -- this is why the default 3-recipe set
    only flips the tree twice, not three times."""
    def show(v):
        return "*" if v is None else ",".join(sorted(v))
    material = f"{ref}|{show(groups)}|{show(states)}"
    return hashlib.sha256(material.encode()).hexdigest()[:16]

def _as_filter(raw: object, field: str, name: str) -> frozenset[str] | None:
    """A missing key means no filter; an empty list means select nothing.

    The distinction is load-bearing -- `upstream` builds unpatched by
    setting `groups = []`, and silently reading that as "no filter" would
    produce a fully patched tree under a name that promises the opposite.
    Also guards against the classic Python footgun: `frozenset("core")`
    silently succeeds as {'c','o','r','e'} since a bare string is iterable.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
        raise RecipeError(f"recipe {name!r}: {field} must be a list of strings")
    return frozenset(raw)


def _options(raw: object, where: str) -> dict[str, str]:
    """cmake options as a flat string->string table. Values are stringified
    rather than typed: they become `-DK=V` on a command line, where TOML's
    bool/int distinction has no meaning and `True` would be the wrong
    spelling of `ON`."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RecipeError(f"{where}: options must be a table")
    out = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            raise RecipeError(
                f"{where}: option {key} is a bool; write \"ON\"/\"OFF\" as a string")
        out[key] = str(value)
    return out


def load_config(path=None) -> Config:
    """Parses recipes.toml. Cross-checks every recipe's builds/platform
    references at LOAD time (not at build time) -- a typo fails when the file
    is read, not after a checkout and patch run have already happened.
    `groups=[]` (empty list) means select nothing; omitting the key means no
    filter (all groups) -- this distinction is load-bearing for `upstream`,
    whose whole purpose is being unpatched."""
    path = Path(path) if path is not None else RECIPES_PATH

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecipeError(f"no recipe file at {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise RecipeError(f"{path}: {exc}") from None

    pinned = raw.get("pinned")
    if not isinstance(pinned, str) or not pinned:
        raise RecipeError(f"{path}: top-level 'pinned' must be a non-empty string")

    builds: dict[str, Build] = {}
    for name, body in (raw.get("build") or {}).items():
        if not isinstance(body, dict):
            raise RecipeError(f"build {name!r} must be a table")
        needs = str(body.get("needs", "none"))
        if needs not in NEEDS:
            raise RecipeError(
                f"build {name!r}: needs must be one of {', '.join(NEEDS)}")
        builds[name] = Build(
            name=name,
            description=str(body.get("description", "")),
            options=_options(body.get("options"), f"build {name!r}"),
            variant_set=body.get("variant-set"),
            needs=needs,
        )

    platforms: dict[str, Platform] = {}
    for name, body in (raw.get("platform") or {}).items():
        if not isinstance(body, dict):
            raise RecipeError(f"platform {name!r} must be a table")
        targets = body.get("targets")
        if not isinstance(targets, str) or not targets:
            raise RecipeError(f"platform {name!r}: targets must be a non-empty string")
        platforms[name] = Platform(
            name=name,
            description=str(body.get("description", "")),
            targets=targets,
            options=_options(body.get("options"), f"platform {name!r}"),
            c_compiler=body.get("c-compiler"),
            cxx_compiler=body.get("cxx-compiler"),
        )

    recipes: dict[str, Recipe] = {}
    for name, body in (raw.get("recipe") or {}).items():
        if not isinstance(body, dict):
            raise RecipeError(f"recipe {name!r} must be a table")
        ref = body.get("ref", FOLLOW_PIN)
        if not isinstance(ref, str) or not ref:
            raise RecipeError(f"recipe {name!r}: ref must be a non-empty string")
        follows = ref == FOLLOW_PIN

        builds_list = body.get("builds")
        if builds_list is None:
            builds_list = []
        elif not isinstance(builds_list, list) or not all(isinstance(v, str) for v in builds_list):
            raise RecipeError(f"recipe {name!r}: builds must be a list of strings")
        for build_name in builds_list:
            if build_name not in builds:
                raise RecipeError(
                    f"recipe {name!r} names unknown build {build_name!r}; "
                    f"choose one of {', '.join(sorted(builds))}")

        platform_name = body.get("platform")
        if platform_name is not None and platform_name not in platforms:
            raise RecipeError(
                f"recipe {name!r} names unknown platform {platform_name!r}; "
                f"choose one of {', '.join(sorted(platforms))}")

        recipes[name] = Recipe(
            name=name,
            ref=pinned if follows else ref,
            groups=_as_filter(body.get("groups"), "groups", name),
            states=_as_filter(body.get("states"), "states", name),
            follows_pin=follows,
            builds=tuple(builds_list),
            platform=platform_name,
            default=bool(body.get("default", False)),
        )

    if not recipes:
        raise RecipeError(f"{path}: defines no recipes")

    return Config(pinned=pinned, recipes=recipes, builds=builds,
                  platforms=platforms, path=path)

def repin(new_ref: str, path=None) -> str:
    """Rewrites ONLY the `pinned = "..."` line in place via regex
    substitution on the raw text -- NOT a full TOML re-serialise, so comments
    and formatting survive. Returns the old value."""
    if path is None:
        path = RECIPES_PATH
    else:
        path = Path(path)

    content = path.read_text(encoding='utf-8')

    # Find current pinned value
    match = re.search(r'^pinned\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise RecipeError(f"could not find pinned = \"...\" in {path}")

    old_ref = match.group(1)
    new_content = re.sub(
        r'^(pinned\s*=\s*)"[^"]+"',
        rf'\1"{new_ref}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    path.write_text(new_content, encoding='utf-8')
    return old_ref

def load(path=None):
    """Load recipes and return (recipes dict, pinned ref)."""
    config = load_config(path)
    return config.recipes, config.pinned

def get(name, path=None) -> Recipe:
    """Get a single recipe by name."""
    config = load_config(path)
    return config.recipe(name)

def names(path=None) -> list[str]:
    """List all recipe names. Returns [] if file is unusable (for --help rendering)."""
    try:
        config = load_config(path)
        return sorted(config.recipes.keys())
    except Exception:
        return []
