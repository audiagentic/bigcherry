# RSC - serving configuration: discovered, derived and tuned, not hand-picked

Goal (owner, 2026-11-01/02): one way to serve a model on whatever hardware is present - easy to run and manage, standard
formats, maximum customisation with minimum complexity, and no hand-picked values. Settings are user/policy intent,
discovered facts, values derived from model shape + profile + facts, or measured/tuned and cached per topology; each
resolved value carries its provenance and `explain` says why it has that value.

| Item | Delivers | Depends on |
| --- | --- | --- |
| [RSC01](RSC01.md) | Typed TOML schema, every setting classified, effective manifest with provenance, `explain`; no serving change | - |
| [RSC02](RSC02.md) | `bigcherry serve` replaces the shell launcher; generated, allowlisted process environments; lifecycle | RSC01 |
| [RSC03](RSC03.md) | One derived start-up resource plan across host RAM, pinned memory and VRAM | RSC01, RSC02 |
| [RSC04](RSC04.md) | Retire the legacy launch surfaces once callers migrate | RSC02, RSC03, RCS10 |

Runs in parallel with the live-placement roadmap (patching-radiance-heat-placement, stage 1 first) and joins it at stage
2: calibration and derived kernel/lane settings stay in build-radiance-components/RCS10 and RHP09; runtime movement
between KV retention, engine experts and helpers stays in RHP13. Every default change passes RHP12, including its
retained-session cohort.

Source: GPT configuration assessment 2026-11-02, reviewed against the code.
