# quant-sweep

Per-card weight-quant comparison for Qwen3.8-27B using one vendor (unsloth) so only the quant
differs. `make-configs.py` writes configs/*.json (3-arm groups sharing an anchor quant);
`queue-quant-sweep.sh` queues them (dual XTX -sm tensor, R9700 alone, one XTX alone). MTP off.
