# DeepSeek-V4-Flash probes

- `probe-v4-flash.sh`: load + layout probe (97 GiB IQ3_XXS; routed experts split between GPUs, CPU RAM and the 6900).
- `queue-v4-flash.sh`: queues it on Brutus (stops/restarts radiance-vllm).
