/* XDP06 native gfx1100 W4A4 group-major metadata research variant.
 * W4A4 arithmetic, A4/W4 inputs, source packed operand layout, model
 * weight distribution and CPU oracle are identical to wide16meta.
 * Only per-group scalar sidecars are packed group-major *once* before H2D.
 * End-to-end quantizer/repack time is NOT measured or implied by kernel times.
 */
#define XDP06_STAGE_META 1
#define XDP06_META_TILE_MAJOR 1
#include "xtx27_w4a4_wide16.hip"
