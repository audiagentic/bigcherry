# tools/lab/strata

Scripts for BCOP37 (qualify the Strata execution architecture for Qwen Flash-Next on the local AMD host).
They run on Brutus, write only into a timestamped result bundle under `/mnt/data/bigcherry-work/runs/bcop37-*`,
and never modify the Strata tree.

- `bcop37-system.sh <bundle>` - step 1, host facts into `system.txt`.
- `bcop37-build.sh <tree> <sha> <gfx> <bundle>` - step 3, clean documented ROCm build for one target with logs.

Owner: BCOP37. Disposition: lab, remove when BCOP37 is closed.
