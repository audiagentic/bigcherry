# Vulkan comparison lab

Plan item: BRVP03 / RRVP05 / PRVP03
Status: active
Owner: bigcherry
Question state: open

## Question

How does the stock Vulkan backend (RADV; AMDVLK once installed) compare with the HIP production build on
Brutus for Qwen3.8-27B and Flash-Next, and what does a Vulkan AllReduce provider buy under -sm tensor?

## Inputs

- `probe-27b-vk.sh`: 27B Q8_0 Vulkan screening (layer/tensor split, MTP, 3-card, single R9700).
- `queue-vk-first.sh`: builds `vulkan-stock:vulkan-stock:vulkan-linux` and runs the screening.

## Outputs

Run outputs under `/mnt/data/bigcherry-work/runs/<run>/` on Brutus.

## Runtime

GPU required: yes
Real compilation required: yes
Mutates canonical BigCherry state: no

## Safety

- Queue jobs only (host + GPU locks, PCIe link preflight); stops/restarts radiance-vllm.
- Do not import this experiment from `bigcherry` production, tests, or maintained analysis.

## Disposition

Open; graduates into RRVP05 campaign tooling.
