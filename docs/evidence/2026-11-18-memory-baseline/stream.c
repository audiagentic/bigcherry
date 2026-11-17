/* Small STREAM-style host memory-bandwidth baseline. */
#define _POSIX_C_SOURCE 200809L
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : 64000000;
    const int repeats = argc > 2 ? atoi(argv[2]) : 7;
    double *a = aligned_alloc(64, n * sizeof(double));
    double *b = aligned_alloc(64, n * sizeof(double));
    double *c = aligned_alloc(64, n * sizeof(double));
    if (!a || !b || !c || repeats < 1) {
        fprintf(stderr, "allocation or argument error\n");
        return 2;
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
    printf("elements=%zu bytes_per_array=%zu threads=%d repeats=%d\n",
           n, n * sizeof(double), omp_get_max_threads(), repeats);
    const char *names[] = {"Copy", "Scale", "Add", "Triad"};
    double best[4] = {0}, total[4] = {0};
    for (int r = 0; r < repeats; ++r) {
        for (int k = 0; k < 4; ++k) {
            double t0 = now();
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                if (k == 0) c[i] = a[i];
                else if (k == 1) b[i] = 3.0 * c[i];
                else if (k == 2) c[i] = a[i] + b[i];
                else a[i] = b[i] + 3.0 * c[i];
            }
            double seconds = now() - t0;
            double gbps = (double)n * sizeof(double) * (k == 2 ? 3.0 : 2.0) / seconds / 1e9;
            if (gbps > best[k]) best[k] = gbps;
            total[k] += gbps;
        }
    }
    for (int k = 0; k < 4; ++k)
        printf("%-5s best_GB_s=%.2f mean_GB_s=%.2f\n", names[k], best[k], total[k] / repeats);
    printf("checksum=%.1f\n", a[n / 2] + b[n / 2] + c[n / 2]);
    free(a); free(b); free(c);
    return 0;
}
