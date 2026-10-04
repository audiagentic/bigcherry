"""0910 (QFP18): named feature sets and built-in help for every BIGCHERRY_* runtime flag in the build.

Feature sets. Patches are screened one at a time behind their own env flags, so a production profile needs many of
them (v6 sets eight). BIGCHERRY_FEATURES=<set>[,<set>...] expands each named set into its member variables from a
load-time constructor in ggml.c (libggml-base loads before any backend or model code reads its flags). A member
that is already set in the environment is left alone, so dev A/B runs can still override one switch
(e.g. BIGCHERRY_ACT_Q81=0). The expansion is logged once to stderr as activation evidence; unknown set names and set
members that no patch in this build documents are reported.

Help. BIGCHERRY_FEATURES=help prints the feature sets and the env-doc table, then exits. Each patch that reads a
runtime flag documents it in its own patch.py as ENV_DOCS = (EnvDoc(...), ...) and requires this patch; the patch
loader turns those into rows inserted before this table's end marker, so the table lists exactly the flags of the
patches in the build.

To add a profile, add one row to bc_feature_sets below; a member may be @other-set. Name sets by scope (what they
act on), not by the model they were tuned on.
"""

from __future__ import annotations

import re

from bigcherry.patcher import ENV_DOC_TABLE_END, Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

# Feature sets: (name, description, members). A member is NAME=VALUE or @other-set. Name sets by what they act on,
# not by the model they were tuned on. Validated below when the patch is loaded; the C table is generated from it.
SETS = (
    ("hip-q81",
     "HIP MMVQ decode: reuse one Q8_1 activation quantization and let RMS-norm, activations, MUL, scale fusions (and "
     "hyper-connection pre-mix, on HC models) write it directly (any quantized model)",
     ("GGML_HIP_Q8_1_CACHE_MODE=on", "BIGCHERRY_RMS_Q81=1", "BIGCHERRY_ACT_Q81=1", "BIGCHERRY_HC_Q81=1",
      "BIGCHERRY_SCALE_ACT_FUSE=1")),
    ("sched-async",
     "scheduler stages small host inputs asynchronously (multi-backend / tensor-split runs)",
     ("BIGCHERRY_SCHED_ASYNC_INPUTS=1",)),
    ("flashnext",
     "Qwen3.8 Flash-Next production profile (pin 0504396); its Qwen4Exp-only patches are on by default",
     ("@hip-q81", "@sched-async")),
)

_SET_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_VAR = re.compile(r"^[A-Z][A-Z0-9_]*=[^ \"\\]+$")


def _validate_sets(sets) -> None:
    """Fail closed on duplicate names, bad tokens, unknown @refs, reference cycles and conflicting assignments."""
    by_name = {}
    for name, desc, members in sets:
        if not _SET_NAME.match(name) or name in by_name:
            raise ValueError(f"0910: bad or duplicate feature set name {name!r}")
        if not desc or '"' in desc or "\\" in desc:
            raise ValueError(f"0910: set {name!r} needs a plain description")
        for m in members:
            if not (m.startswith("@") and _SET_NAME.match(m[1:])) and not _VAR.match(m):
                raise ValueError(f"0910: set {name!r} has a bad member {m!r}")
        by_name[name] = members

    def flatten(name, stack):
        if name in stack:
            raise ValueError(f"0910: feature set cycle {' -> '.join((*stack, name))}")
        out = {}
        for m in by_name[name]:
            if m.startswith("@"):
                if m[1:] not in by_name:
                    raise ValueError(f"0910: set {name!r} references unknown set {m!r}")
                items = flatten(m[1:], (*stack, name)).items()
            else:
                items = [tuple(m.split("=", 1))]
            for var, value in items:
                if out.get(var, value) != value:
                    raise ValueError(f"0910: set {name!r} assigns {var} both {out[var]!r} and {value!r}")
                out[var] = value
        return out

    for name in by_name:
        flatten(name, ())


def _c_sets_table(sets) -> str:
    rows = "".join(f'    {{ "{n}", "{d}",\n      "{" ".join(m)}" }},\n' for n, d, m in sets)
    return "static const char * const bc_feature_sets[][3] = {\n    // name, description, members (generated)\n" + rows + "};\n"


_validate_sets(SETS)

_A = "#include <signal.h>\n"
_N = r"""#include <signal.h>

// bigcherry 0910: named feature sets (BIGCHERRY_FEATURES=<set>[,<set>...] | help); explicit member variables win
""" + _c_sets_table(SETS) + r"""
typedef struct {
    const char * name;
    const char * patch;
    const char * values;
    const char * def;
    const char * desc;
} bc_env_doc;

// one block per patch in this build that reads a runtime flag (its ENV_DOCS, added by the patch loader)
static const bc_env_doc bc_env_docs[] = {
    { "BIGCHERRY_FEATURES", "0910_feature_sets", "<set>[,<set>...] | help", "unset",
      "enable named feature sets; 'help' prints the sets and every documented flag, then exits" },
""" + ENV_DOC_TABLE_END + r"""};

static int bc_env_documented(const char * name) {
    for (size_t i = 0; i < sizeof(bc_env_docs)/sizeof(bc_env_docs[0]); i++) {
        if (strcmp(bc_env_docs[i].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static void bc_feature_setenv(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

static void bc_feature_help(void) {
    fprintf(stderr, "BigCherry feature sets (BIGCHERRY_FEATURES=<set>[,<set>...]; explicit member variables win):\n");
    for (size_t i = 0; i < sizeof(bc_feature_sets)/sizeof(bc_feature_sets[0]); i++) {
        fprintf(stderr, "  %s - %s\n", bc_feature_sets[i][0], bc_feature_sets[i][1]);
        char members[1024];
        snprintf(members, sizeof(members), "%s", bc_feature_sets[i][2]);
        for (char * m = members; *m != '\0'; ) {
            char * end = strchr(m, ' ');
            char * next = end ? end + 1 : m + strlen(m);
            if (end) {
                *end = '\0';
            }
            char name[128];
            snprintf(name, sizeof(name), "%s", m);
            char * eq = strchr(name, '=');
            if (eq) {
                *eq = '\0';
            }
            fprintf(stderr, "      %s%s\n", m,
                    m[0] == '@' || bc_env_documented(name) ? "" : "  (no patch in this build documents it)");
            m = next;
        }
    }
    fprintf(stderr, "\nBigCherry runtime flags in this build:\n");
    for (size_t i = 0; i < sizeof(bc_env_docs)/sizeof(bc_env_docs[0]); i++) {
        const bc_env_doc * d = &bc_env_docs[i];
        fprintf(stderr, "  %s = %s (default: %s) [%s]\n      %s\n", d->name, d->values, d->def, d->patch, d->desc);
    }
}

// expand one set into its variables (and the sets it references), appending " NAME=VALUE" items to applied
static int bc_feature_set_apply(const char * set, char * applied, size_t cap, int depth) {
    for (size_t i = 0; i < sizeof(bc_feature_sets)/sizeof(bc_feature_sets[0]); i++) {
        if (strcmp(set, bc_feature_sets[i][0]) != 0) {
            continue;
        }
        char members[1024];
        snprintf(members, sizeof(members), "%s", bc_feature_sets[i][2]);
        for (char * m = members; *m != '\0'; ) {
            char * end = strchr(m, ' ');
            char * next = end ? end + 1 : m + strlen(m);
            if (end) {
                *end = '\0';
            }
            if (m[0] == '@') {
                if (depth >= 8 || !bc_feature_set_apply(m + 1, applied, cap, depth + 1)) {
                    fprintf(stderr, "BIGCHERRY_FEATURES set '%s' references unknown or too deeply nested set '%s'\n", set, m + 1);
                }
                m = next;
                continue;
            }
            char * eq = strchr(m, '=');
            if (eq == NULL) {
                m = next;
                continue;
            }
            *eq = '\0';
            const int preset = getenv(m) != NULL;
            if (!preset) {
                bc_feature_setenv(m, eq + 1);
            }
            char item[192];
            snprintf(item, sizeof(item), " %s=%s%s%s", m, preset ? getenv(m) : eq + 1, preset ? "(explicit)" : "",
                     bc_env_documented(m) ? "" : "(not in build)");
            strncat(applied, item, cap - strlen(applied) - 1);
            m = next;
        }
        return 1;
    }
    return 0;
}

static void bc_feature_sets_expand(void) {
    const char * req = getenv("BIGCHERRY_FEATURES");
    if (req == NULL || req[0] == '\0') {
        return;
    }
    if (strcmp(req, "help") == 0 || strcmp(req, "list") == 0) {
        bc_feature_help();
        exit(0);
    }
    char sets[256];
    if (strlen(req) >= sizeof(sets)) {  // fail closed: never enable part of a request
        fprintf(stderr, "BIGCHERRY_FEATURES too long (%zu >= %zu chars): ignored entirely\n", strlen(req), sizeof(sets));
        return;
    }
    snprintf(sets, sizeof(sets), "%s", req);
    for (char * tok = strtok(sets, ","); tok != NULL; tok = strtok(NULL, ",")) {
        while (*tok == ' ') {
            tok++;
        }
        char applied[1024] = "";
        if (bc_feature_set_apply(tok, applied, sizeof(applied), 0)) {
            fprintf(stderr, "BIGCHERRY_FEATURES %s:%s\n", tok, applied);
        } else {
            fprintf(stderr, "BIGCHERRY_FEATURES unknown set '%s' (ignored; BIGCHERRY_FEATURES=help lists sets)\n", tok);
        }
    }
}

// Explicit, idempotent entry point: called from ggml_init's first call and the backend registry constructor (so it
// runs on every compiler, including MSVC, before backends or models read their flags), and from a load-time
// constructor where the compiler supports one (earliest, before other libraries' static initialisers read flags).
GGML_API void ggml_bigcherry_features_init(void);
void ggml_bigcherry_features_init(void) {
    static int done = 0;
    ggml_critical_section_start();
    const int first = !done;
    done = 1;
    ggml_critical_section_end();
    if (first) {
        bc_feature_sets_expand();
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void bc_feature_sets_ctor(void) {
    ggml_bigcherry_features_init();
}
#endif
"""

_A_INIT = ("        // initialize time system (required on Windows)\n"
           "        ggml_time_init();\n")
_N_INIT = _A_INIT + "        ggml_bigcherry_features_init();  // bigcherry 0910: feature sets before any flag is read\n"

_A_REG_INC = "#include \"ggml-impl.h\"\n"
_N_REG_INC = _A_REG_INC + "extern \"C\" void ggml_bigcherry_features_init(void);  // bigcherry 0910 (ggml.c)\n"
_A_REG = "    ggml_backend_registry() {\n"
_N_REG = _A_REG + "        ggml_bigcherry_features_init();  // bigcherry 0910: before any backend registers or reads its flags\n"

PATCHES = [
    FilePatch(
        path="ggml/src/ggml.c",
        description="0910: BIGCHERRY_FEATURES named feature sets + env-flag help table",
        language="none",
        edits=(
            Edit(id="feature-sets", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 0910: named feature sets", rationale="ggml.c standard include block (signal.h).",
                 expect_matches=1, max_span_lines=2),
            Edit(id="feature-sets-ggml-init", anchor=re.escape(_A_INIT), mode="replace", text=_N_INIT,
                 guard=r"bigcherry 0910: feature sets before any flag is read", rationale="ggml_init first-call block.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-reg.cpp",
        description="0910: apply feature sets before the backend registry registers any backend",
        language="none",
        edits=(
            Edit(id="feature-sets-reg-decl", anchor=re.escape(_A_REG_INC), mode="replace", text=_N_REG_INC,
                 guard=r"bigcherry 0910 \(ggml\.c\)", rationale="ggml-backend-reg.cpp include block.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="feature-sets-reg-ctor", anchor=re.escape(_A_REG), mode="replace", text=_N_REG,
                 guard=r"bigcherry 0910: before any backend registers", rationale="ggml_backend_registry constructor.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]
