#!/bin/bash
# Native llama.cpp (llama-native:stock:linux-multi = pristine c061, no overlay/patches, same RCCL HIP build)
# vs the BigCherry production build, on the same probes and layouts: 27B Q8_0 (dual-XTX tensor/layer,
# MTP5, 3-card, sidecar on the 6900, single R9700) and Flash-Next (baseline, best 8K MTP config, 192K). 3 timed
# requests per layout. Stops radiance-vllm for the run and restarts it.
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
docker stop radiance-vllm >/dev/null 2>&1 && docker update --restart always radiance-vllm >/dev/null
jobs=$(mktemp)
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-native-c061 llama-native:stock:linux-multi - gfx1100,gfx1201,gfx1030
VIS=0,1,2,3 SCRIPT native-27b tools/lab/dflash/probe-27b.sh @b-native-c061 /mnt/data/bigcherry-work/runs/native-27b/out (plain|mtp5|mtp5-3card|plain-layer|mtp5-layer|mtpside5-d6900|dt-mtp5-a|dt-mtp5-b|single-r9700|single-r9700-mtp5) 3
VIS=0,1,2,3 SCRIPT prod-27b tools/lab/dflash/probe-27b.sh @b-flash-c061 /mnt/data/bigcherry-work/runs/prod-27b/out (plain|mtp5|mtp5-3card|plain-layer|mtp5-layer|mtpside5-d6900|dt-mtp5-a|dt-mtp5-b|single-r9700|single-r9700-mtp5) 3
VIS=0,1,2,3 SCRIPT native-flash tools/lab/flash-next/layout-probe.sh @b-native-c061 /mnt/data/bigcherry-work/runs/native-flash/out (cpu3-tensor|cpu3-tensor-ub2048|8k-443-n3-ub1024-d6900|192k-223-n3-ub512b-d6900)
JOBS
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
echo ALL_JOBS_DONE
