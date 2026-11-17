#!/bin/bash
cd <repo>; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
export RADIANCE_HOME=<work-root>/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3/radiance_home BC_SERVE_BINARY=<work-root>/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3/radiance_home/bin/bc-serve RAD_BIN=<work-root>/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3/bin/radiance RAD_HOME=<work-root>/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3/radiance_home R9700_BDF=0000:09:00.0 KLD_REF_DIR=<work-root>/runs/rad116-kld/reference
PYTHONPATH=tools timeout 5400 python3 -m bigcherry.campaign.benchmark --server-config <work-root>/runs/rhp20-tail-kld/config.json --output <work-root>/runs/rhp20-tail-kld/out --pairs 1 > <work-root>/runs/rhp20-tail-kld/kld.log 2>&1
echo $? > <work-root>/runs/rhp20-tail-kld/exit
