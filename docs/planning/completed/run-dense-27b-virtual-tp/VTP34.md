---
id: VTP34
order: 34
plan: run-dense-27b-virtual-tp
state: superseded
created-at: '2026-11-12T07:39:02.019970+00:00'
breadth: ''
skill: advanced
created-by: agent
work: L
priority: P1
---

# Finish the move to the manifest: generic-configuration audit C01-C24 (bigcherry-1 / bigcherry-2 split)

## Description

Owner rule (non-negotiable): the engine and tooling are generic. Models, layouts and topologies, splits, helper configurations, expert placement and kernel choices are configuration (manifest fields, run-time layout files, model metadata). There are no predefined setups in code, no build per model or configuration, and no code duplicated per model. Radiance upstream is the baseline.

The audit (GPT, 2026-11-12) is in `GENERIC-CONFIG-AUDIT.md` in this plan: findings C01-C24 with status, plus plan P0-P4.

## Steps

1. bigcherry-2: C11, C15, C16, C17 in one PR off main; hardware identity of a Flash-Next helper layout and qwen27b-tp3 through the launcher.
2. bigcherry-1: P0 and C09. P0 is planned as RCS13 (C02) then RCS14 (C05); staging in RCS12.
3. Then C03, C06/C14, C13, C18, C19-C22, each as its own PR.
4. Extend the model-constant guard to topology bounds and named defaults.

## Detailed Solution & Technical Design



## Code Samples & Guidance



## Files



## Validation

- The audit re-run reports no open finding.
- Launcher identity runs (output byte-identical to before) for helper and engine-only layouts.

## Effort & Risk



## Standards



## Acceptance Criteria

- Every audit finding is fixed or explicitly accepted, with a reason.
- A new model, split, topology or helper configuration needs only data (layout or manifest) and no code change or rebuild.

## Notes

Coordination and status: /instances/.teams/bigcherry/TASKS.md.

2026-11-26 roadmap reconciliation at 165d6f724: Superseded by native YAML RSC11 (and RSC01/02/03/05/06/12, PWR07/08), RCS12/RCS10 for model geometry/defaults, PA49 for conditional independent llama.cpp C22. All 24 findings have disposition: C01/02/05/10/18/20 RCS12 (C05 residual RAD23); C03 completed RUM08 rad0013 retired; C04/07/12/23/24 prior fixed evidence; C06 explicitly superseded by RSP01 immutable compiled archs.toml; C08/11/13/15/16/17/19 RSC11; C09/14 RCS10 and RSC06; C21 PWR07/08; C22 PA49 conditional. Audit GENERIC-CONFIG-AUDIT.md retained in original group, all previous work/evidence unchanged. Supersession is transfer, not proof that outstanding acceptance passed. No duplicate manifest implementation.

## Done (PR #863)

- Run-time layouts (`--layouts` / `BIGCHERRY_LAYOUTS`, examples in engines/radiance/layouts/) with no default layout.
- Mixed-architecture engine groups; layout `[settings]` with resolver keys in both resolvers.
- Contract amendments 18 and 19; the P1 producer accepts dense models.
- One build for every topology (the VTP14 patch set in the main source).
- C04, C07, C12, C23 and C24 fixed.

## Split (agreed with bigcherry-1, 2026-11-12)

- **bigcherry-2:** C11 (layout and recipe `server-args` become `[settings]`; the launcher sets only options given explicitly), C15 (Python consumes `bc-serve --explain` instead of re-resolving), C16 (no Flash-Next default in the launcher), C17 (lab tools take layouts and experiments as inputs). Separate PR off main. The launcher's gpu-lease and power start/stop stay as they are, or go to bigcherry-1.
- **bigcherry-1:** P0 (C01, C02, C05, C10: remote-MoE protocol shapes from the negotiated LayoutShape; C10 merged as #897) and C09 (rad0008 RAD_HC_UPMIX_RT default 128 against upstream 256: A/B).
- **Later:** C03 (MTP replay geometry), C06 and C14 (archs, calibration and helper-profile seeds at run time), C13 (diagnostic builds as runtime switches), C18 (shared ownership operations across plugins), C19-C22.

## Change Log

- 2026-11-12T07:39:02.019970+00:00 (created-by): Created by agent
- 2026-11-12T07:40:21.954735+00:00 (updated-by): Updated: section:description, section:steps, section:validation, section:acceptance_criteria, section:notes
- 2026-11-25T19:02:23.021227+00:00 (updated-by): Updated: section:notes
- 2026-11-25T19:02:27.840945+00:00 (state-transition): State: pending → superseded
