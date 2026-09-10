# HI168 E2E evidence manifest

- schema: 1
- plan: HI168
- observed: 2026-09-07/08 (Brutus, configured build server)
- runner: maintained `bigcherry ab-benchmark` with `server-bench`
- roles: stock / BC native / BC replay
- schedule: six balanced order blocks, 18 cells per run
- result: complete exploratory matrix; physical-device admission blocker retained

Raw receipt SHA-256 digests:

| receipt | SHA-256 |
| --- | --- |
| `27b-dual-three-arm-001/run.json` | `9CEBFC1E8FBA8C92C225B0A0C49D9FD124CDA2DD37D145ACD68242FBE2578F02` |
| `9b-gpu0-three-arm-001/run.json` | `AF104D6B3FC3718053B41D7938C156737152FCC16D7B8E8D9461B922563E78FB` |
| `9b-gpu1-three-arm-001/run.json` | `BBC9B33E0C69E4187E3E53402B4441A192B7F43B9A858B64A8D24925AF94C3A7` |
| `9b-gpu2-three-arm-001/run.json` | `C49AC59DCD9E1AB423A9CC72BBA7D49454EED6DC3E436BBA1231C103A0C5B273` |
| `9b-gpu3-three-arm-001/run.json` | `B7C74EAC4B26F926A8B9AD6D5DF4179A1C9C97DBAB9EC3F3DE7B852D6EA6C580` |

The raw directory is intentionally ignored by source control; this tracked
manifest and the compact report are the durable evidence index.
