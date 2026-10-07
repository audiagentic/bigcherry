## What this slice does

Promotes 1329_alloc_top_trace and 1331_alloc_peak_live to validated and adds them to the production validated-enhancements set. Both remain inert unless their diagnostic environment variables are set. Experiment entries used to build them separately are removed or stripped of the now-production diagnostic.

## Hardware evidence

build deploy-v6-alloc-peak compiled clean; with BIGCHERRY_ALLOC_PEAK=1 BIGCHERRY_ALLOC_TOP=1 the server log has 1298 ALLOC_PEAK and 1106 ALLOC_TOP lines; greedy text identical to the arm without them (md5 a34a34c1f48d both), 0 error lines.

## Verified / not verified

Verified: supplied Brutus compile, activation-line counts, flag-on/off greedy identity and zero-error smoke result.

Not verified here: no new GPU run was possible from the GitHub-only environment; no performance claim is made for these diagnostics.
