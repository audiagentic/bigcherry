## Scope

Adds the b11474 MET01 real-case qualification harness for 1337/1338: host-resident routed experts (`--n-cpu-moe N`), 0 MiB versus 4096 MiB cache, then the same 4096 MiB cache with `BIGCHERRY_MOE_CACHE_PROFILE`. The runner reports prefill t/s, decode t/s, MTP acceptance and greedy md5 per request/arm. RV4222 is closed by restoring its still-applicable graph-safe adaptive-cache qualification gate into current MET01.

## Hardware evidence already supplied

build moe-cache-profile compiled clean and is byte-identical in output with the flag off (smoke md5 equals production).

## Verified / not verified

Verified: the supplied b11474 compile and flag-off smoke identity above.

Not verified here: the new 0/4096/profile GPU sweep has not yet been run on Brutus; no cache-performance or profile-correctness claim is made by this slice.
