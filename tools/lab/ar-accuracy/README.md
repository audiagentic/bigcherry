# ar-accuracy

Accuracy gates for lossy AllReduce wire formats (bf16/f16/q8_0 host wire, adaptive provider)
before any of them can become a production default. `kld.sh` runs llama-perplexity
--kl-divergence against an exact RCCL f32 reference; `gates.py` applies the gates (mean KLD <=
0.001, p99 KLD <= 0.01, same top token >= 99.5%) and checks MTP acceptance equality across A/B
arms (within 0.5 pp). `queue-ar-accuracy.sh` queues the whole set on the dual XTX.
