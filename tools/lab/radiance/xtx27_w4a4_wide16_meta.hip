/* XDP06: isolated gfx1100 native IU4 group-header LDS ablation.
 * Uses unchanged W4A4 arithmetic and output mapping from wide16.
 * Each group preloads A/W group scales and code-sums/zero-points
 * cooperatively once rather than repeatedly from global per output.
 * Build depends on this source and its included wide16 source.
 */
#define XDP06_STAGE_META 1
#include "xtx27_w4a4_wide16.hip"
