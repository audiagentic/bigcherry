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

To add a profile, add one row to bc_feature_sets below.
"""

from __future__ import annotations

import re

from bigcherry.patcher import ENV_DOC_TABLE_END, Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

_A = "#include <signal.h>\n"
_N = r"""#include <signal.h>

// bigcherry 0910: named feature sets (BIGCHERRY_FEATURES=<set>[,<set>...] | help); explicit member variables win
static const char * const bc_feature_sets[][3] = {
    // name, description, members (NAME=VALUE, space separated)
    { "flashnext-v6", "Qwen3.8 Flash-Next production profile v6 (pin 0504396)",
      "GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 "
      "BIGCHERRY_HC_Q81=1 BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1 BIGCHERRY_QSA_HOST_REMAP=1" },
};

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
            fprintf(stderr, "      %s%s\n", m, bc_env_documented(name) ? "" : "  (no patch in this build documents it)");
            m = next;
        }
    }
    fprintf(stderr, "\nBigCherry runtime flags in this build:\n");
    for (size_t i = 0; i < sizeof(bc_env_docs)/sizeof(bc_env_docs[0]); i++) {
        const bc_env_doc * d = &bc_env_docs[i];
        fprintf(stderr, "  %s = %s (default: %s) [%s]\n      %s\n", d->name, d->values, d->def, d->patch, d->desc);
    }
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
    snprintf(sets, sizeof(sets), "%s", req);
    for (char * tok = strtok(sets, ","); tok != NULL; tok = strtok(NULL, ",")) {
        while (*tok == ' ') {
            tok++;
        }
        int found = 0;
        for (size_t i = 0; i < sizeof(bc_feature_sets)/sizeof(bc_feature_sets[0]); i++) {
            if (strcmp(tok, bc_feature_sets[i][0]) != 0) {
                continue;
            }
            found = 1;
            char members[1024];
            snprintf(members, sizeof(members), "%s", bc_feature_sets[i][2]);
            char applied[1024] = "";
            for (char * m = members; *m != '\0'; ) {
                char * end = strchr(m, ' ');
                char * next = end ? end + 1 : m + strlen(m);
                if (end) {
                    *end = '\0';
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
                strncat(applied, item, sizeof(applied) - strlen(applied) - 1);
                m = next;
            }
            fprintf(stderr, "BIGCHERRY_FEATURES %s:%s\n", tok, applied);
        }
        if (!found) {
            fprintf(stderr, "BIGCHERRY_FEATURES unknown set '%s' (ignored; BIGCHERRY_FEATURES=help lists sets)\n", tok);
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void bc_feature_sets_ctor(void) {
    bc_feature_sets_expand();
}
#endif
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml.c",
        description="0910: BIGCHERRY_FEATURES named feature sets + env-flag help table",
        language="none",
        edits=(
            Edit(id="feature-sets", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 0910: named feature sets", rationale="ggml.c standard include block (signal.h).",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]
