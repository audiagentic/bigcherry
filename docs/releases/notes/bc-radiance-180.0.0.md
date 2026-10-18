# bc-radiance-180.0.0

BigCherry on radiance `v1.8.0` (`7a7707a95559`), BigCherry commit `fc4567a6ef73`.
Previous release: `bc-radiance-1.7.0-r0`.

## Bump record

pin-bump lane build: radiance:stock:linux-radiance build_plan_id=a1e0e5abea7fe9cd973f70a1400824ee binary_sha256=1ca6a08bda08ec5fe278fb1fa5d6824f4b7d87ad667766a1e6cefa7311245e9f build_dir=/mnt/vault/development/projects/bigcherry/workspaces/main/work/builds/2608d4b2f6e47b979d16fffc3b489842/a1e0e5abea7fe9cd973f70a1400824ee

release 180.0.0 2026-10-19
upstream: radiance v1.8.0 (revision 7a7707a95559c0b85277e75b7799fcf2ca8b8bc5)

completion gate (pin-status --complete --all-remotes):
pin-status: local
  pin           v1.8.0 -> 7a7707a95559
  vendor HEAD   7a7707a95559 (tag v1.8.0)  [0 modified, 0 untracked]
  releases      0504396140d1 audited, 0adcc3bb5710 broken, 1e968f6e1d86 patched, 22dc605c4ead patched, 23b0202a189c patched, 2578138397d7 generated, 28ff0958291c patched, 370cb12e8bb7 patched, 434ddbbc0e30 patched, 50182a53fa2c generated, 6ea215d171fd patched, 7a7707a95559 patched, 91f6a6cf3613 patched, abc123 validated, b1ff4ca23630 patched, b9acf138a1e2 patched, c061df19838f broken, c841aeeb8bb2 patched, d77dd0806dc2 patched, d7bd3bfcad3e patched, d89651a7b205 patched
  bigcherry     62a0348ed8af
  marker        7a7707a95559 -> 7a7707a95559 (v1.8.0) [committed-clean]
  VERDICT       consistent

COMPLETION  PASS

hardware evidence:
{
  "schema": 1,
  "engine": "radiance",
  "started": "2026-10-19T03:15:32+1100",
  "passed": true,
  "entries": [
    {
      "name": "helper-ab",
      "config": "tools/lab/radiance/helper-ab.json",
      "output": "/mnt/vault/development/projects/bigcherry/workspaces/main/work/accept-180c-031532/helper-ab/run",
      "returncode": 0,
      "error": "",
      "passed": true,
      "baselines": [
        {
          "file": "docs/reference/engines/radiance/baselines/helper-ab.json",
          "pin": "1.7.0",
          "run": "arm heat-hot = helper-ab rows0 with hot-first ownership (rad06-heat-10172150, Brutus, radiance 1.7.0 natives on, 2 rounds)",
          "date": "2026-10-17"
        }
      ],
      "rows": [
        {
          "entry": "helper-ab",
          "arm": "rows0",
          "metric": "pp4096_tps",
          "baseline": 3311.0,
          "measured": 3389.085,
          "limit": 3211.67,
          "direction": "higher-better",
          "passed": true,
          "detail": "",
          "delta": 78.08500000000004
        },
        {
          "entry": "helper-ab",
          "arm": "rows0",
          "metric": "pp1024_tps",
          "baseline": 3320.0,
          "measured": 3439.175,
          "limit": 3220.4,
          "direction": "higher-better",
          "passed": true,
          "detail": "",
          "delta": 119.17500000000018
        },
        {
          "entry": "helper-ab",
          "arm": "rows0",
          "metric": "tg512_tps",
          "baseline": 143.0,
          "measured": 145.56,
          "limit": 135.85,
          "direction": "higher-better",
          "passed": true,
          "detail": "",
          "delta": 2.5600000000000023
        },
        {
          "entry": "helper-ab",
          "arm": "rows0",
          "metric": "tg2048_tps",
          "baseline": 150.0,
          "measured": 152.905,
          "limit": 147.0,
          "direction": "higher-better",
          "passed": true,
          "detail": "",
          "delta": 2.905000000000001
        }
      ]
    },
    {
      "name": "helper-kld",
      "config": "tools/lab/radiance/helper-kld.json",
      "output": "/mnt/vault/development/projects/bigcherry/workspaces/main/work/accept-180c-031532/helper-kld/run",
      "returncode": 0,
      "error": "",
      "passed": true,
      "baselines": [
        {
          "file": "docs/reference/engines/radiance/baselines/helper-kld.json",
          "pin": "1.7.0",
          "run": "rad06-kld-hot-10172208 (Brutus, radiance 1.7.0 natives on, hot-first ownership, helper-kld.json vs the 1.3.0 two-XTX exact reference)",
          "date": "2026-10-17"
        }
      ],
      "rows": [
        {
          "entry": "helper-kld",
          "arm": "helper-on",
          "metric": "kld_mean",
          "baseline": 0.0633,
          "measured": 0.06142075525045341,
          "limit": 0.07329999999999999,
          "direction": "lower-better",
          "passed": true,
          "detail": "",
          "delta": -0.0018792447495465867
        },
        {
          "entry": "helper-kld",
          "arm": "helper-on",
          "metric": "kld_p99",
          "baseline": 0.673,
          "measured": 0.6242077606916407,
          "limit": 0.773,
          "direction": "lower-better",
          "passed": true,
          "detail": "",
          "delta": -0.0487922393083593
        },
        {
          "entry": "helper-kld",
          "arm": "helper-on",
          "metric": "top1_percent",
          "baseline": 88.3,
          "measured": 89.08980582524272,
          "limit": 87.3,
          "direction": "higher-better",
          "passed": true,
          "detail": "",
          "delta": 0.7898058252427234
        }
      ]
    }
  ]
}

## Patches

### New patches

- `rad0005_mp4_ffmpeg6_writer` (untested) - Let radiance 1.8.0's MP4 writer build against FFmpeg 6.
