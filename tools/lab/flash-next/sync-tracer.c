// LD_PRELOAD shim: counts hipStreamSynchronize call sites (RNX01: 465 host syncs per MTP decode step).
// Wraps the HIP runtime entry, hashes a short backtrace per call, and at exit prints the top stacks with
// symbols (dynamic symbols only; enough for libllama/libggml exports). Counting starts when the file named by
// SYNC_TRACER_ARM exists, so model load/warm-up is excluded; the client creates it right before the timed decode.
// Build: gcc -O2 -shared -fPIC sync-tracer.c -o sync-tracer.so -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEPTH 8
#define SLOTS 4096
typedef int (*sync_fn)(void *);
struct slot { uint64_t hash; unsigned long count; void * frames[DEPTH]; int n; };
static struct slot slots[SLOTS];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long total;

static int armed(void) {
    static int on = 0;
    static unsigned long probe = 0;
    if (on) return 1;
    if ((probe++ & 255) != 0) return 0;  // check the arm file every 256 calls
    const char * f = getenv("SYNC_TRACER_ARM");
    if (f && access(f, F_OK) == 0) on = 1;
    return on;
}

int hipStreamSynchronize(void * stream) {
    static sync_fn real = NULL;
    if (!real) real = (sync_fn) dlsym(RTLD_NEXT, "hipStreamSynchronize");
    if (armed()) {
        void * fr[DEPTH + 1];
        int n = backtrace(fr, DEPTH + 1) - 1;  // drop this frame
        uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < n; i++) { h ^= (uint64_t) fr[i + 1]; h *= 1099511628211ull; }
        pthread_mutex_lock(&mu);
        total++;
        for (unsigned i = h % SLOTS, k = 0; k < SLOTS; i = (i + 1) % SLOTS, k++) {
            if (slots[i].count == 0) { slots[i].hash = h; slots[i].n = n; memcpy(slots[i].frames, fr + 1, n * sizeof(void *)); }
            if (slots[i].hash == h) { slots[i].count++; break; }
        }
        pthread_mutex_unlock(&mu);
    }
    return real(stream);
}

static int cmp(const void * a, const void * b) {
    const struct slot * x = a, * y = b;
    return x->count < y->count ? 1 : x->count > y->count ? -1 : 0;
}

__attribute__((destructor)) static void dump(void) {
    const char * out = getenv("SYNC_TRACER_OUT");
    FILE * f = out ? fopen(out, "w") : stderr;
    if (!f) return;
    qsort(slots, SLOTS, sizeof(slots[0]), cmp);
    fprintf(f, "hipStreamSynchronize calls while armed: %lu\n", total);
    for (int i = 0; i < 12 && slots[i].count; i++) {
        fprintf(f, "== %lu calls\n", slots[i].count);
        char ** s = backtrace_symbols(slots[i].frames, slots[i].n);
        for (int j = 0; s && j < slots[i].n; j++) fprintf(f, "   %s\n", s[j]);
        free(s);
    }
    if (f != stderr) fclose(f);
}
