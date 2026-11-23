---
id: VTP55
order: 0
plan: run-dense-27b-virtual-tp
state: pending
created-at: '2026-11-23T21:26:04.398434+00:00'
breadth: ''
skill: advanced
created-by: agent
work: M
priority: P1
---

# flashnext-4gpu RDNA2 tail helper: expert GEMM spills and occupancy, and the per-request fixed cost (about 190 us a layer)

## Description

The RDNA2 tail helper (gfx1030) paces the flashnext-4gpu MoE collective (VTP54): 871 us a request against 548-559 on the XTX helpers although it carries a third of the routes. Stage times per request (large lane): gate/up 478 us against 252, reduce 99 against 51, routes 121 against 72, quant 18 against 12, down level (158 against 162). Fitting the two request sizes in the helper logs (about 5 and 92 rows a request) gives a fixed cost of about 186 us a request on the tail (105 us on the XTX) plus 7.5 us a row (4.8): the fixed part times 49 layers is about 9.1 ms a pass on the tail.

**Static census of the tail's kernels (libr2, release 2020.8.5).** Wave32; no matrix unit; no dot-product instruction in any kernel (the expert GEMMs are f32 fused-multiply-add loops, 1045-2084 static FMAs a kernel, on dequantised 4-bit weights and E4M3 activations). The expert GEMM instantiations use 80-256 VGPRs; every instantiation at 80-128 VGPRs spills (100-864 B private memory a thread: eight of eight at 112 / 128, six at 192, the 80-96 family at 392-536 B), and the spill-free ones at 193-256 VGPRs run only four to five waves per SIMD. The small kernels (routes, quant, norm) are 7-94 VGPRs without spills.

**What we will try, in order.** (1) From the helper's launch structure, count kernels and launches per request on the tail (routes, gate/up, quant, down, reduce) and which stages are serialised by host or stream synchronisation; the fixed ~190 us is a launch-and-sync number, not an arithmetic one. (2) Resource and occupancy table of the instantiations the tail actually launches at 1-8 and 9-32 rows (the helper's kernel selection), and a spill fix where the selected one spills (the same method as the IQR drafter: split the live state, keep the arithmetic order, bit-identical). (3) A dot-product form for the int8-activation expert GEMM (gfx1030 has the packed int8 dot) is evaluated only after (1) and (2): at decode row counts the request is dominated by its fixed part, which a faster inner loop does not touch.

## Steps

1. Launch count and sync points per request on the tail from the helper source (static), plus the helper's lane stage timers; table of stage x launches.
2. Resource table for the selected kernels at the real row counts; flag spills and waves per SIMD.
3. If a selected kernel spills: scratch fix with byte identity against the shipped kernel on the host reference, cold timing on the tail card at 1-32 rows; then the traced run of VTP54 decides the step effect.
4. Per-request fixed cost: fuse or overlap the small stages (routes, quant, reduce) only where the launch table shows serialisation; byte identity gate.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- Byte identity of expert outputs against the host reference and the shipped kernel at every tested row count.
- Resource record before / after; cold timing on the tail card; flashnext-4gpu decode ms/step ABBA (3 seeds) for any change that ships.

## Effort & Risk

Medium. The tail card is used only by this layout; gains there reach the step only while the tail is the pacer, which VTP54 measures.

## Standards



## Acceptance Criteria

- The launch and sync table is committed; a change ships only with a measured flashnext-4gpu step gain above the seed spread; otherwise a NO-GO with the table.

## Notes

Ceiling basis (2026-11-24): the 6.8 ms gap is a per-request model fitted from whole-run helper averages (two request sizes, prefill included) against the engine rank's mailbox wait; it is a compute ceiling under stated assumptions, not a measured bound, and the wait is not separable from transport with the data we have (the lab trace is not reachable from a production start). The libr2 waves-per-SIMD change is judged by its own resource table, cold replay on the tail card and a flashnext-4gpu step A/B, not by this model.



**2026-11-24 closed NO-GO on the GEMM and placement side.** (1) The waves-per-SIMD cap experiment (a constexpr rule for gfx10.3) was bit-identical but slower in 26 of 54 cold cases, and the source audit shows production decode on the tail never reaches the grouped expert GEMM that carried the spills: helper rows 1-8 use the dedicated GEMV kernels and 9-32 the dedicated few-rows kernels. (2) The helper's own kernels (gfx1030, 2020.8.5) are spill-free: GEMV forms 31-114 VGPRs, the few-rows forms 77-184 VGPRs, wave32, most using the packed dot instruction. (3) Placement is derived from capacity and heat, not from a speed ratio: the two XTX helpers take candidate experts in heat order until each VRAM budget is full and the tail takes the remainder from the cold end (9546 + 9546 + 5986 of 25078; total budget 59.5 GiB against 59.3 GiB of experts, so the cards are full), leaving the tail 24% of the experts and 14% of the routes. (4) Stage times, tail against XTX (large lane): routes 1.7x, gate/up 1.9x, quant 1.5x, down 1.0x, reduce 1.9x, against the 1.875x nominal memory-bandwidth ratio of the two cards: the tail sits at its card's memory and latency class. What remains is the per-request floor (about 190 us against 105 us), a launch-chain question tracked separately as a bounded replay before any brief.



**2026-11-24 split verdict.** REJECT the waves-per-SIMD rule for the RDNA2 helper (bit-identical but slower in 26 of 54 cold cases; production never reaches the kernel). ACCEPT-WORK, ranked and not started: a fused front end for the helper request (the four route kernels and the activation preparation as one or two launches; about 0.5-0.7 ms a pass; risk: it changes the helper request path and needs byte identity on routes), after the larger items.

## Change Log

- 2026-11-23T21:26:04.398434+00:00 (created-by): Created by agent
- 2026-11-23T21:26:33.858915+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:effort_risk, section:acceptance_criteria
- 2026-11-23T22:51:54.858796+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:11:26.103416+00:00 (updated-by): Updated: section:notes
- 2026-11-23T23:21:57.000342+00:00 (updated-by): Updated: section:notes
