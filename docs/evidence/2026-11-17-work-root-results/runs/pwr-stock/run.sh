#!/bin/bash
cd <repo>
export RADIANCE_HOME=<work-root>/builds/daf793f13be8d5762deecb73e3372641/41f15405cc38031fe6591674f460b8f1/radiance_home BC_SERVE_BINARY=<work-root>/builds/daf793f13be8d5762deecb73e3372641/41f15405cc38031fe6591674f460b8f1/radiance_home/bin/bc-serve
bash <work-root>/runs/pwr-stock/pwr01-copy.sh 60 > <work-root>/runs/pwr-stock/out.log 2>&1; echo "rc=$?" > <work-root>/runs/pwr-stock/status
