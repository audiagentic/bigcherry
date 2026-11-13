/* Optional, read-only Radiance MoE histogram observer. No pointer is a wire value.
 * The counts pointer is owned by Ctx and expires when the synchronous callback returns.
 * Callback runs on the engine's CPU rank thread; it must not block, retain counts,
 * issue device work, or throw. Concurrent ranks may call the same observer concurrently.
 * The producer never changes routing, heat policy, or credit on callback failure.
 */
#ifndef RAD_MOE_HEAT_H
#define RAD_MOE_HEAT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { RAD_MOE_HEAT_ABI_V1 = 1 };
enum RadMoeHeatPhaseV1 {
    RAD_MOE_HEAT_PREFILL = 1,
    RAD_MOE_HEAT_DECODE = 2,
    RAD_MOE_HEAT_MIXED = 3,
    RAD_MOE_HEAT_DRAFT = 4,
    RAD_MOE_HEAT_UNKNOWN = 5
};
enum RadMoeHeatFlagsV1 {
    RAD_MOE_HEAT_EVENT_READY = 1u << 0,
    RAD_MOE_HEAT_SOURCE_INCOMPLETE = 1u << 1,
    RAD_MOE_HEAT_PASS_PHASE_MIXED = 1u << 2
};
enum RadMoeHeatRegisterStatusV1 {
    RAD_MOE_HEAT_REGISTER_OK = 0,
    RAD_MOE_HEAT_REGISTER_BAD_ABI = -1,
    RAD_MOE_HEAT_REGISTER_OCCUPIED = -2,
    RAD_MOE_HEAT_REGISTER_BAD_ARGUMENT = -3
};

typedef struct RadMoeHeatRowV1 {
    uint32_t abi_version, struct_size;
    uint64_t run_generation;       /* process-local Ctx lifecycle; pair with launch ID in transport */
    uint64_t engine_step;          /* source step at issue, never the drain step */
    uint64_t bank_generation;      /* one generation per ring reuse, 0 is invalid */
    uint64_t dropped_source_total; /* Ctx route-ring drop watermark (not transport loss) */
    uint32_t rank, layer;          /* rank-local producer; never assume TP ranks are identical */
    uint32_t expert_base, n_expert;/* model-global expert IDs [base,base+n_expert) */
    uint32_t pass_reports;         /* last-bank histogram weighted by this multiplicity */
    uint32_t phase;                /* captured at report issue, not callback time */
    uint32_t flags;                /* ready/incomplete/same-step heterogeneous phases */
    uint32_t reserved;             /* must be zero */
    const int32_t* counts;         /* borrowed, read only; callback lifetime only */
} RadMoeHeatRowV1;

/* Return 0 on acceptance. Errors are counted but do not veto source credit. */
typedef int (*RadMoeHeatObserverV1)(void* user, const RadMoeHeatRowV1* row);

/* Explicit exact ABI/size negotiation; NULL unregisters and drains in-flight
 * callbacks before the caller may free user memory. Do not unregister from
 * inside the callback. Only one observer per engine process in V1. */
int rad_moe_heat_register_v1(uint32_t abi_version, uint32_t row_size,
                             RadMoeHeatObserverV1 callback, void* user);
/* Process-local rejected/failed callbacks; includes malformed source records. */
uint64_t rad_moe_heat_callback_errors_v1(void);

#ifdef __cplusplus
}
#endif
#endif
