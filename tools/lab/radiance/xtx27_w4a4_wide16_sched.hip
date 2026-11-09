/* XDP06: independent gfx1100 wide16 scheduler ablation.
 * This translation unit only enables the compile-time fence in the existing
 * BigCherry wide16 proof. The runtime still uses the full numerical oracle,
 * and the wrapper fingerprints BOTH this file and the included source.
 * Production Radiance remains untouched.
 */
#define XDP06_SCHED_FENCE 1
#include "xtx27_w4a4_wide16.hip"
