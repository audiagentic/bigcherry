"""0910 (QFP18/QFP23): runtime profiles from the release profile file + built-in help for every runtime flag.

Profiles. Model- and scope-specific runtime settings are not compiled into patch code: they live in
the profile/ folder (*.ini), which ships next to the binaries (canonical source: the overlay's src/profile/; the build
copies it into the runtime output directory and installs it with the binaries). BIGCHERRY_FEATURES=<p>[,<p>]
applies the named profiles: each profile lists NAME = VALUE flags and may include other profiles (@name). Explicit
environment variables win, so A/B runs can still override one switch. The file is found relative to libggml-base
(dladdr / GetModuleHandleEx), or at BIGCHERRY_PROFILES=<folder|file>.

Application is all-or-nothing: the request and the file are parsed and flattened (unknown profiles, include cycles,
conflicting assignments, malformed lines are errors) before the first variable is set; a failed set rolls back. The
library never exits: BIGCHERRY_FEATURES=help prints the profiles and the env-doc table and returns
GGML_BIGCHERRY_FEATURES_HELP; llama-server maps HELP to exit 0 and ERROR to exit 2.

ggml_bigcherry_features_init() is idempotent and runs before any flag is read: from ggml_init, before the backend
registry is constructed, before backends are dlopened/scored, from llama-server's main, and (GCC/Clang) from a
load-time constructor. Explicit hooks make it work on MSVC too.

Help table. Each patch that reads a runtime flag documents it in its own patch.py as ENV_DOCS = (EnvDoc(...), ...)
and requires this patch; the patch loader turns those into rows inserted before this table's end marker, so the table
lists exactly the flags of the patches in the build. tools/bigcherry/patch/runtime_profiles.py validates the profile
files against those ENV_DOCS in patch-lint.
"""

from __future__ import annotations

import re

from bigcherry.patcher import ENV_DOC_TABLE_END, Edit, FilePatch

GROUP = "rdna-boosts"
STATE = "validated"

PROFILE_DIR = "profile"

_A = "#include <signal.h>\n"
_N = r"""#include <signal.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN  // same settings ggml.c uses for its own <windows.h> include below
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#endif

// bigcherry 0910: runtime profiles (BIGCHERRY_FEATURES=<profile>[,...] | help) from the profile/ folder
enum { GGML_BIGCHERRY_FEATURES_ERROR = -1, GGML_BIGCHERRY_FEATURES_OK = 0, GGML_BIGCHERRY_FEATURES_HELP = 1 };

typedef struct {
    const char * name;
    const char * patch;
    const char * values;
    const char * def;
    const char * desc;
} bc_env_doc;

// one block per patch in this build that reads a runtime flag (its ENV_DOCS, added by the patch loader)
static const bc_env_doc bc_env_docs[] = {
    { "BIGCHERRY_FEATURES", "0910_feature_sets", "<profile>[,<profile>...] | help", "unset",
      "apply runtime profiles from the profile/ folder; 'help' prints the profiles and every documented flag" },
    { "BIGCHERRY_PROFILES", "0910_feature_sets", "<folder|file>", "profile/ next to libggml-base",
      "runtime profile folder (every *.ini) or single file to load instead of the one shipped with the binaries" },
""" + ENV_DOC_TABLE_END + r"""};

enum { BC_MAX_PROFILES = 64, BC_MAX_ITEMS = 1024, BC_MAX_PENDING = 128, BC_MAX_REQ = 4096, BC_ARENA = 65536 };

typedef struct { int profile; int kind; const char * a; const char * b; } bc_item;  // kind 0: NAME=VALUE, 1: @include
typedef struct { const char * name; const char * desc; const char * arch; int line; } bc_profile;

static char       bc_arena[BC_ARENA];
static size_t     bc_arena_used;
static bc_profile bc_profiles[BC_MAX_PROFILES];
static int        bc_n_profiles;
static bc_item    bc_items[BC_MAX_ITEMS];
static int        bc_n_items;
static char       bc_profile_path[1024];

static const char * bc_store(const char * b, size_t n) {
    if (bc_arena_used + n + 1 > BC_ARENA) {
        return NULL;
    }
    char * p = bc_arena + bc_arena_used;
    memcpy(p, b, n);
    p[n] = '\0';
    bc_arena_used += n + 1;
    return p;
}

static int bc_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

static void bc_trim(const char ** b, const char ** e) {
    while (*b < *e && bc_ws(**b)) (*b)++;
    while (*e > *b && bc_ws((*e)[-1])) (*e)--;
}

static int bc_is_name(const char * b, const char * e) {  // profile name: [a-z0-9][a-z0-9-]*, <= 63
    if (b == e || e - b > 63 || !((*b >= 'a' && *b <= 'z') || (*b >= '0' && *b <= '9'))) return 0;
    for (const char * p = b; p < e; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '-')) return 0;
    }
    return 1;
}

static int bc_is_flag(const char * b, const char * e) {  // flag: [A-Z][A-Z0-9_]*, <= 127
    if (b == e || e - b > 127 || !(*b >= 'A' && *b <= 'Z')) return 0;
    for (const char * p = b; p < e; p++) {
        if (!((*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '_')) return 0;
    }
    return 1;
}

static int bc_find_profile(const char * b, const char * e) {
    for (int i = 0; i < bc_n_profiles; i++) {
        if (strlen(bc_profiles[i].name) == (size_t) (e - b) && memcmp(bc_profiles[i].name, b, (size_t) (e - b)) == 0) {
            return i;
        }
    }
    return -1;
}

// locate the profiles: BIGCHERRY_PROFILES (a folder or one file), else the profile/ folder next to this library
static int bc_locate_profiles(void) {
    const char * env = getenv("BIGCHERRY_PROFILES");
    if (env != NULL && env[0] != '\0') {
        snprintf(bc_profile_path, sizeof(bc_profile_path), "%s", env);
        return strlen(env) < sizeof(bc_profile_path);
    }
    char lib[1024] = "";
#if defined(_WIN32)
    HMODULE mod = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            (LPCSTR) (void *) &bc_locate_profiles, &mod) ||
        GetModuleFileNameA(mod, lib, (DWORD) sizeof(lib)) == 0) {
        return 0;
    }
#else
    Dl_info info;
    if (dladdr((void *) &bc_locate_profiles, &info) == 0 || info.dli_fname == NULL) {
        return 0;
    }
    snprintf(lib, sizeof(lib), "%s", info.dli_fname);
#endif
    char * slash = strrchr(lib, '/');
#if defined(_WIN32)
    char * bslash = strrchr(lib, '\\');
    if (bslash != NULL && (slash == NULL || bslash > slash)) slash = bslash;
#endif
    const size_t dir_len = slash ? (size_t) (slash - lib + 1) : 0;
    if (dir_len + strlen("profile") + 1 > sizeof(bc_profile_path)) return 0;
    memcpy(bc_profile_path, lib, dir_len);
    snprintf(bc_profile_path + dir_len, sizeof(bc_profile_path) - dir_len, "%s", "profile");
    return 1;
}

// parse one profile file into the tables (grammar: tools/bigcherry/patch/runtime_profiles.py); 1 = ok, -1 = invalid
static int bc_parse_profile_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) return -1;
    char line[1024];
    int n = 0, cur = -1, ok = 1;
    while (ok && fgets(line, sizeof(line), f) != NULL) {
        n++;
        if (strchr(line, '\n') == NULL && !feof(f)) { ok = 0; break; }  // line too long
        const char * b = line;
        const char * e = line + strlen(line);
        bc_trim(&b, &e);
        if (b == e || *b == '#') continue;
        if (*b == '[') {
            if (e[-1] != ']' || bc_n_profiles >= BC_MAX_PROFILES) { ok = 0; break; }
            const char * nb = b + 1;
            const char * ne = e - 1;
            bc_trim(&nb, &ne);
            if (!bc_is_name(nb, ne) || bc_find_profile(nb, ne) >= 0) { ok = 0; break; }
            cur = bc_n_profiles++;
            bc_profiles[cur].name = bc_store(nb, (size_t) (ne - nb));
            bc_profiles[cur].desc = "";
            bc_profiles[cur].arch = "";
            bc_profiles[cur].line = n;
            ok = bc_profiles[cur].name != NULL;
            continue;
        }
        if (cur < 0 || bc_n_items >= BC_MAX_ITEMS) { ok = 0; break; }
        if (*b == '@') {
            const char * rb = b + 1;
            const char * re = e;
            bc_trim(&rb, &re);
            if (!bc_is_name(rb, re)) { ok = 0; break; }
            bc_items[bc_n_items++] = (bc_item) { cur, 1, bc_store(rb, (size_t) (re - rb)), NULL };
            ok = bc_items[bc_n_items - 1].a != NULL;
            continue;
        }
        const char * eq = memchr(b, '=', (size_t) (e - b));
        if (eq == NULL) { ok = 0; break; }
        const char * kb = b;
        const char * ke = eq;
        const char * vb = eq + 1;
        const char * ve = e;
        bc_trim(&kb, &ke);
        bc_trim(&vb, &ve);
        if (vb == ve) { ok = 0; break; }
        if (ke - kb == 11 && memcmp(kb, "description", 11) == 0) {
            bc_profiles[cur].desc = bc_store(vb, (size_t) (ve - vb));
            ok = bc_profiles[cur].desc != NULL;
        } else if (ke - kb == 4 && memcmp(kb, "arch", 4) == 0) {
            bc_profiles[cur].arch = bc_store(vb, (size_t) (ve - vb));  // last arch wins in help; Python checks all
            ok = bc_profiles[cur].arch != NULL;
        } else if (bc_is_flag(kb, ke)) {
            for (const char * p = vb; p < ve; p++) {
                if (bc_ws(*p) || *p == '"' || *p == '\\' || *p == '#') { ok = 0; break; }
            }
            if (ok) {
                bc_items[bc_n_items++] = (bc_item) { cur, 0, bc_store(kb, (size_t) (ke - kb)), bc_store(vb, (size_t) (ve - vb)) };
                ok = bc_items[bc_n_items - 1].a != NULL && bc_items[bc_n_items - 1].b != NULL;
            }
        } else {
            ok = 0;
        }
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "BIGCHERRY_FEATURES: %s:%d: invalid profile line; no profile applied\n", path, n);
        return -1;
    }
    return 1;
}

static int bc_has_ini_suffix(const char * name) {
    const size_t n = strlen(name);
    return n > 4 && strcmp(name + n - 4, ".ini") == 0;
}

// load every *.ini of the profile folder (or the one file BIGCHERRY_PROFILES names); 1 = ok, 0 = none, -1 = invalid
static int bc_load_profiles(void) {
    if (!bc_locate_profiles()) return 0;
    char file[1300];
    int files = 0;
#if defined(_WIN32)
    snprintf(file, sizeof(file), "%s\\*.ini", bc_profile_path);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(file, &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && bc_has_ini_suffix(fd.cFileName)) {
                snprintf(file, sizeof(file), "%s\\%s", bc_profile_path, fd.cFileName);
                if (bc_parse_profile_file(file) < 0) { FindClose(h); return -1; }
                files++;
            }
        } while (FindNextFileA(h, &fd));
        FindClose(h);
        return files > 0 ? 1 : 0;
    }
#else
    DIR * dir = opendir(bc_profile_path);
    if (dir != NULL) {
        struct dirent * de;
        while ((de = readdir(dir)) != NULL) {
            if (de->d_name[0] == '.' || !bc_has_ini_suffix(de->d_name)) continue;
            snprintf(file, sizeof(file), "%s/%s", bc_profile_path, de->d_name);
            if (bc_parse_profile_file(file) < 0) { closedir(dir); return -1; }
            files++;
        }
        closedir(dir);
        return files > 0 ? 1 : 0;
    }
#endif
    FILE * single = fopen(bc_profile_path, "rb");  // BIGCHERRY_PROFILES may name one file
    if (single == NULL) return 0;
    fclose(single);
    return bc_parse_profile_file(bc_profile_path);
}

typedef struct { const char * name; const char * value; int set_by_us; } bc_pending;

// flatten one profile into pending (include cycle and conflicting assignment are errors)
static int bc_flatten(int p, bc_pending * pending, int * np, int * stack, int depth) {
    for (int d = 0; d < depth; d++) {
        if (stack[d] == p) {
            fprintf(stderr, "BIGCHERRY_FEATURES: include cycle through [%s]; no profile applied\n", bc_profiles[p].name);
            return -1;
        }
    }
    if (depth >= BC_MAX_PROFILES) return -1;
    stack[depth] = p;
    for (int i = 0; i < bc_n_items; i++) {
        const bc_item * it = &bc_items[i];
        if (it->profile != p) continue;
        if (it->kind == 1) {
            const int q = bc_find_profile(it->a, it->a + strlen(it->a));
            if (q < 0) {
                fprintf(stderr, "BIGCHERRY_FEATURES: [%s] includes unknown @%s; no profile applied\n", bc_profiles[p].name, it->a);
                return -1;
            }
            if (bc_flatten(q, pending, np, stack, depth + 1) != 0) return -1;
            continue;
        }
        int found = 0;
        for (int j = 0; j < *np; j++) {
            if (strcmp(pending[j].name, it->a) != 0) continue;
            found = 1;
            if (strcmp(pending[j].value, it->b) != 0) {
                fprintf(stderr, "BIGCHERRY_FEATURES: %s set to both %s and %s; no profile applied\n", it->a, pending[j].value, it->b);
                return -1;
            }
        }
        if (!found) {
            if (*np >= BC_MAX_PENDING) return -1;
            pending[(*np)++] = (bc_pending) { it->a, it->b, 0 };
        }
    }
    return 0;
}

static int bc_env_set(const char * name, const char * value) {
#if defined(_WIN32)
    return (int) _putenv_s(name, value);
#else
    return setenv(name, value, 0);
#endif
}

static void bc_env_unset(const char * name) {
#if defined(_WIN32)
    (void) _putenv_s(name, "");
#else
    (void) unsetenv(name);
#endif
}

static void bc_feature_help(int loaded) {
    if (loaded > 0) {
        fprintf(stderr, "BigCherry runtime profiles (%s; BIGCHERRY_FEATURES=<profile>[,...]; explicit variables win):\n", bc_profile_path);
        for (int p = 0; p < bc_n_profiles; p++) {
            fprintf(stderr, "  %s%s%s - %s\n", bc_profiles[p].name, bc_profiles[p].arch[0] ? " [arch " : "",
                    bc_profiles[p].arch[0] ? bc_profiles[p].arch : "", bc_profiles[p].desc);
            if (bc_profiles[p].arch[0]) fprintf(stderr, "      (arch %s)\n", bc_profiles[p].arch);
            for (int i = 0; i < bc_n_items; i++) {
                if (bc_items[i].profile != p) continue;
                if (bc_items[i].kind == 1) fprintf(stderr, "      @%s\n", bc_items[i].a);
                else fprintf(stderr, "      %s=%s\n", bc_items[i].a, bc_items[i].b);
            }
        }
    } else {
        fprintf(stderr, "BigCherry runtime profiles: no profile file (%s)\n", loaded == 0 ? "not found" : "invalid");
    }
    fprintf(stderr, "\nBigCherry runtime flags in this build:\n");
    for (size_t i = 0; i < sizeof(bc_env_docs)/sizeof(bc_env_docs[0]); i++) {
        const bc_env_doc * d = &bc_env_docs[i];
        fprintf(stderr, "  %s = %s (default: %s) [%s]\n      %s\n", d->name, d->values, d->def, d->patch, d->desc);
    }
}

static int bc_feature_sets_expand(void) {
    const char * raw = getenv("BIGCHERRY_FEATURES");
    if (raw == NULL || raw[0] == '\0') return GGML_BIGCHERRY_FEATURES_OK;
    const size_t len = strlen(raw);
    if (len > BC_MAX_REQ) {
        fprintf(stderr, "BIGCHERRY_FEATURES: request longer than %d bytes; no profile applied\n", BC_MAX_REQ);
        return GGML_BIGCHERRY_FEATURES_ERROR;
    }
    const char * begin = raw;
    const char * end = raw + len;
    bc_trim(&begin, &end);
    const int loaded = bc_load_profiles();
    if ((end - begin == 4 && memcmp(begin, "help", 4) == 0) || (end - begin == 4 && memcmp(begin, "list", 4) == 0)) {
        bc_feature_help(loaded);
        return GGML_BIGCHERRY_FEATURES_HELP;
    }
    if (loaded <= 0) {
        fprintf(stderr, "BIGCHERRY_FEATURES: profile file %s (%s); no profile applied\n",
                bc_profile_path[0] ? bc_profile_path : "profile/", loaded == 0 ? "not found" : "invalid");
        return GGML_BIGCHERRY_FEATURES_ERROR;
    }
    bc_pending pending[BC_MAX_PENDING];
    int np = 0;
    int requested[BC_MAX_PROFILES];
    int nr = 0;
    int stack[BC_MAX_PROFILES];
    for (const char * p = begin; ; ) {
        const char * q = p;
        while (q < end && *q != ',') q++;
        const char * tb = p;
        const char * te = q;
        bc_trim(&tb, &te);
        const int idx = bc_is_name(tb, te) ? bc_find_profile(tb, te) : -1;
        if (idx < 0) {
            fprintf(stderr, "BIGCHERRY_FEATURES: unknown or malformed profile '%.*s' (BIGCHERRY_FEATURES=help lists them); "
                            "no profile applied\n", (int) (te - tb), tb);
            return GGML_BIGCHERRY_FEATURES_ERROR;
        }
        for (int i = 0; i < nr; i++) {
            if (requested[i] == idx) {
                fprintf(stderr, "BIGCHERRY_FEATURES: profile '%s' requested twice; no profile applied\n", bc_profiles[idx].name);
                return GGML_BIGCHERRY_FEATURES_ERROR;
            }
        }
        requested[nr++] = idx;
        if (bc_flatten(idx, pending, &np, stack, 0) != 0) return GGML_BIGCHERRY_FEATURES_ERROR;
        if (q == end) break;
        p = q + 1;
    }
    // whole request validated: apply, explicit variables win, roll back on failure
    char applied[2048] = "";
    for (int i = 0; i < np; i++) {
        char item[256];
        if (getenv(pending[i].name) != NULL) {
            snprintf(item, sizeof(item), " %s=<explicit>", pending[i].name);
        } else {
            if (bc_env_set(pending[i].name, pending[i].value) != 0) {
                fprintf(stderr, "BIGCHERRY_FEATURES: failed to set %s; rolled back, no profile applied\n", pending[i].name);
                for (int j = 0; j < i; j++) {
                    if (pending[j].set_by_us) bc_env_unset(pending[j].name);
                }
                return GGML_BIGCHERRY_FEATURES_ERROR;
            }
            pending[i].set_by_us = 1;
            snprintf(item, sizeof(item), " %s=%s", pending[i].name, pending[i].value);
        }
        if (strlen(applied) + strlen(item) < sizeof(applied)) strcat(applied, item);
    }
    fprintf(stderr, "BIGCHERRY_FEATURES %.*s (%s):%s\n", (int) (end - begin), begin, bc_profile_path, applied);
    return GGML_BIGCHERRY_FEATURES_OK;
}

// Explicit, idempotent entry point (see the patch docstring for where it is called). Returns the status of the one
// expansion: GGML_BIGCHERRY_FEATURES_OK / _HELP / _ERROR.
GGML_API int ggml_bigcherry_features_init(void);
int ggml_bigcherry_features_init(void) {
    static int done = 0;
    static int status = GGML_BIGCHERRY_FEATURES_OK;
    ggml_critical_section_start();
    if (!done) {
        status = bc_feature_sets_expand();
        done = 1;
    }
    const int result = status;
    ggml_critical_section_end();
    return result;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void bc_feature_sets_ctor(void) {
    (void) ggml_bigcherry_features_init();  // early fast path; never exits
}
#endif
"""

_A_INIT = "struct ggml_context * ggml_init(struct ggml_init_params params) {\n"
_N_INIT = _A_INIT + "    (void) ggml_bigcherry_features_init();  // bigcherry 0910: runtime profiles before any flag is read\n"

_A_REG_INC = "#include \"ggml-impl.h\"\n"
_N_REG_INC = _A_REG_INC + "extern \"C\" int ggml_bigcherry_features_init(void);  // bigcherry 0910 (ggml.c)\n"
_A_GET_REG = ("static ggml_backend_registry & get_reg() {\n"
              "    static ggml_backend_registry reg;\n")
_N_GET_REG = ("static ggml_backend_registry & get_reg() {\n"
              "    (void) ggml_bigcherry_features_init();  // bigcherry 0910: before the registry registers any backend\n"
              "    static ggml_backend_registry reg;\n")
_A_LOAD_ALL = "void ggml_backend_load_all_from_path(const char * dir_path) {\n"
_N_LOAD_ALL = _A_LOAD_ALL + "    (void) ggml_bigcherry_features_init();  // bigcherry 0910: before any backend is dlopened / scored\n"

_A_MAIN = ("int main(int argc, char ** argv) {\n"
           "    return llama_server(argc, argv);\n")
_N_MAIN = ("extern \"C\" int ggml_bigcherry_features_init(void);  // bigcherry 0910 (ggml.c)\n"
           "\n"
           "int main(int argc, char ** argv) {\n"
           "    switch (ggml_bigcherry_features_init()) {  // bigcherry 0910: BIGCHERRY_FEATURES=help -> 0, invalid -> 2\n"
           "        case 1:  return 0;\n"
           "        case -1: return 2;\n"
           "        default: break;\n"
           "    }\n"
           "    return llama_server(argc, argv);\n")

_A_CMAKE = "add_subdirectory(src)\n"
_N_CMAKE = r"""# bigcherry 0910: ship the runtime profile/ folder next to the binaries (build tree and install)
file(GLOB BIGCHERRY_PROFILE_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/profile/*.ini")
if (BIGCHERRY_PROFILE_FILES)
    set(BIGCHERRY_PROFILE_OUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    if (NOT BIGCHERRY_PROFILE_OUT_DIR)
        set(BIGCHERRY_PROFILE_OUT_DIR "${CMAKE_BINARY_DIR}/bin")
    endif()
    set(BIGCHERRY_PROFILE_OUTPUTS "")
    foreach(_bc_profile ${BIGCHERRY_PROFILE_FILES})
        get_filename_component(_bc_name "${_bc_profile}" NAME)
        add_custom_command(
            OUTPUT  "${BIGCHERRY_PROFILE_OUT_DIR}/profile/${_bc_name}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${BIGCHERRY_PROFILE_OUT_DIR}/profile"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_bc_profile}" "${BIGCHERRY_PROFILE_OUT_DIR}/profile/${_bc_name}"
            DEPENDS "${_bc_profile}"
            COMMENT "bigcherry 0910: runtime profile ${_bc_name}")
        list(APPEND BIGCHERRY_PROFILE_OUTPUTS "${BIGCHERRY_PROFILE_OUT_DIR}/profile/${_bc_name}")
    endforeach()
    add_custom_target(bigcherry_profiles ALL DEPENDS ${BIGCHERRY_PROFILE_OUTPUTS})
    add_dependencies(ggml-base bigcherry_profiles)
    install(FILES ${BIGCHERRY_PROFILE_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR}/profile)
endif()

add_subdirectory(src)
"""

PATCHES = [
    FilePatch(
        path="ggml/src/ggml.c",
        description="0910: runtime profiles loader + env-flag help table + idempotent init",
        language="none",
        edits=(
            Edit(id="feature-sets", anchor=re.escape(_A), mode="replace", text=_N,
                 guard=r"bigcherry 0910: runtime profiles \(BIGCHERRY_FEATURES", rationale="ggml.c standard include block (signal.h).",
                 expect_matches=1, max_span_lines=2),
            Edit(id="feature-sets-ggml-init", anchor=re.escape(_A_INIT), mode="replace", text=_N_INIT,
                 guard=r"bigcherry 0910: runtime profiles before any flag is read", rationale="First line of ggml_init.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="ggml/src/ggml-backend-reg.cpp",
        description="0910: apply runtime profiles before the backend registry exists and before backends are loaded",
        language="none",
        edits=(
            Edit(id="feature-sets-reg-decl", anchor=re.escape(_A_REG_INC), mode="replace", text=_N_REG_INC,
                 guard=r"bigcherry 0910 \(ggml\.c\)", rationale="ggml-backend-reg.cpp include block.",
                 expect_matches=1, max_span_lines=2),
            Edit(id="feature-sets-get-reg", anchor=re.escape(_A_GET_REG), mode="replace", text=_N_GET_REG,
                 guard=r"bigcherry 0910: before the registry registers any backend", rationale="get_reg(), before the static registry.",
                 expect_matches=1, max_span_lines=3),
            Edit(id="feature-sets-load-all", anchor=re.escape(_A_LOAD_ALL), mode="replace", text=_N_LOAD_ALL,
                 guard=r"bigcherry 0910: before any backend is dlopened", rationale="ggml_backend_load_all_from_path entry.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
    FilePatch(
        path="tools/server/main.cpp",
        description="0910: llama-server maps BIGCHERRY_FEATURES help/error to exit codes",
        language="none",
        edits=(
            Edit(id="feature-sets-server-main", anchor=re.escape(_A_MAIN), mode="replace", text=_N_MAIN,
                 guard=r"bigcherry 0910: BIGCHERRY_FEATURES=help -> 0", rationale="llama-server main.",
                 expect_matches=1, max_span_lines=3),
        ),
    ),
    FilePatch(
        path="CMakeLists.txt",
        description="0910: copy the profile/ folder next to the binaries and install it",
        language="none",
        edits=(
            Edit(id="feature-sets-cmake", anchor=re.escape(_A_CMAKE), mode="replace", text=_N_CMAKE,
                 guard=r"bigcherry 0910: ship the runtime profile/ folder", rationale="Root CMakeLists, after ggml is added.",
                 expect_matches=1, max_span_lines=2),
        ),
    ),
]
