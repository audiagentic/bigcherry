#!/bin/bash
cd <repo>
export RADIANCE_HOME=<work-root>/builds/67a435be11abe6053872355915df70a4/e315c3373c1f397eb16273556a7509ab/radiance_home BC_SERVE_BINARY=<work-root>/builds/67a435be11abe6053872355915df70a4/e315c3373c1f397eb16273556a7509ab/radiance_home/bin/bc-serve
bash <work-root>/runs/pwr-stock/pwr01-old.sh 60 > <work-root>/runs/pwr-stock/outold.log 2>&1; echo "rc=$?" > <work-root>/runs/pwr-stock/statusold
