# PKC01 representation sufficiency review

Date: 2026-09-10  
Decision: existing EC16/EC19 representation is sufficient; no new lifecycle
registry or runtime family is required.

## Scope checked

PKC01 asks whether non-GEMM work can be represented without forcing it into
the five matmul dispatch families. The review covered the completed EC16
target-classification schema, EC19 computed lifecycle status, and the current
contracts for the representative FLASH_ATTN and GDN work owned by RD04-RD06
and RD50 (with RD51-RD53 explicitly subsumed by RD50's atomic patch).

| Concern | Existing representation | Evidence | Result |
| --- | --- | --- | --- |
| Runtime matmul family identity | `hypothesis.family` / `target.kind = kernel_family`, constrained to the five `FAMILIES` values | `tools/bigcherry/experiment/contract.py`; RD08 contract | Preserved; not used for non-GEMM work |
| Flash attention classification | `target.kind = attention`; no `target.family` | RD04, RD05, RD06 contracts in `config/experiment-contracts.toml` | Sufficient |
| GDN/SSM classification | `target.kind = ssm_gdn`; no `target.family` | RD50 contract; RD51-RD53 are documented as subsumed in `atomic_part` | Sufficient |
| Graph fusion classification | `target.kind = graph_fusion` | RD13 contract | Sufficient |
| Topology/orchestration classification | `target.kind = tp_topology` / `orchestration` | RD20, RD19, RD39-44 contracts | Sufficient |
| Scope and trigger | `scope.backend`, `scope.architectures`, positive/control workloads and models | RD04-RD06 and RD50 contract sections | Sufficient and orthogonal to lifecycle |
| Correctness requirements | `correctness` table (`backend_reference`, `ppl_equality`, `bit_identical`, etc.) | RD04-RD06 and RD50 | Sufficient; no new field needed |
| Lifecycle status | EC19 computed status signals, independent of target kind | `tools/bigcherry/experiment/contract.py` and EC19 | Correct ownership; do not extend EC19 |

## Decision

The representation gap that created KC01 has already been resolved by EC16's
orthogonal `[target]` section. The contracts preserve the five runtime
matmul families while allowing non-GEMM classifications and their existing
owners' correctness/performance gates. No implementation change is required.

Future gaps must be filed under the owning backend/provider or contract item;
they must not add a parallel family registry, result key, or lifecycle system.
