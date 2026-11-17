#!/usr/bin/env python3
"""CPU-only reproduction of RCS13 Run 6 strict-v2 owner/footprint divergence.

Inputs are the source-matched completed baseline build and preserved failed
manifest directory. No GPU helper, engine process, or measured manifest is run.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for key in ('source_root','build','heat','site','budget','measuring','helper_log'):
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args()
    src=a.source_root.resolve()
    build=a.build.resolve()
    bc=build/'radiance_home/bin/bc-serve'
    assert bc.is_file() and a.site.is_file() and a.heat.is_file()
    assert hashlib.sha256(a.heat.read_bytes()).hexdigest()==(
        'cbf8e17ad4e0e57470da2c000904d863621d5affcf8876c9b87960b9472ba06e'
    ), 'heat changed'
    env=os.environ.copy()
    env.update({'BIGCHERRY_LAYOUTS':str(src/'engines/radiance/layouts'),
                'BC_MODEL_ROOT':str(Path(env.get('BC_MODEL_ROOT','/models'))),
                'BC_WORK_ROOT':str(Path(env.get('BC_WORK_ROOT','/tmp/bigcherry'))),
                'BIGCHERRY_ENVIRONMENT':str(Path(env['BIGCHERRY_ENVIRONMENT']))})
    with tempfile.TemporaryDirectory(prefix='rcs13-owner-cpu-') as td:
        out=Path(td)/'owner-map.json'
        cmd=[str(bc),'--write-manifest',str(out),'--unmeasured',
             '--site',str(a.site),'--power-profile','stock',
             '--set','layout.name=flashnext-tp2xtx-r9700',
             '--set','service.max_sessions=1','--set','service.max_context=8192']
        proc=subprocess.run(cmd,text=True,capture_output=True,env=env,timeout=90)
        assert proc.returncode==0,(proc.returncode,proc.stderr)
        cpu=json.loads(out.read_text())['spec']['plan']['owner_map']
    budget=json.loads(a.budget.read_text())
    measuring=json.loads(a.measuring.read_text())['spec']
    meas_owner=measuring['plan']['owner_map']
    assert budget['host_pool_mib']==32005, budget
    assert cpu['owner_map_sha256']==budget['owner_map_digest']
    assert cpu['owner_map_sha256']!=meas_owner['owner_map_sha256']
    pre_count=[h['count'] for h in cpu['helpers']]
    measure_count=[h['count'] for h in meas_owner['helpers']]
    assert pre_count==[12269] and measure_count==[12641],(pre_count,measure_count)
    shm=(src/'engines/radiance/components/moe-remote/moe_remote_shm.h').read_text()
    def c(name):
        return int(re.search(r'constexpr uint32_t '+name+r'\s*=\s*(\d+)\s*;',shm).group(1))
    max_rows,chunk=c('kMaxRows'),c('kCombineChunkRows')
    configured=measuring['handoff']['transport']['max_rows']
    slots=measuring['processes']['engine']['scheduler']['tp']
    assert measuring['processes']['helper_common']['runtime']['power_save']
    assert len(measuring['processes']['helpers'])==1
    expected=(((max_rows+chunk-1)//chunk)-((configured+chunk-1)//chunk))*4*slots
    log=a.helper_log.read_text()
    match=re.search(r'prepared lanes take (\d+) bytes, the shared footprint says (\d+)',log)
    assert match,'runtime helper footprint mismatch missing'
    observed=int(match.group(1))-int(match.group(2))
    assert expected==observed==120,(expected,observed)
    print(json.dumps({'cpu_only':True,'host_pool_mib':budget['host_pool_mib'],
      'preflight_owner_sha':cpu['owner_map_sha256'],
      'measuring_owner_sha':meas_owner['owner_map_sha256'],
      'preflight_counts':pre_count,'measuring_counts':measure_count,
      'predicted_excess_bytes':expected,'observed_excess_bytes':observed},indent=2))
if __name__=='__main__': main()
