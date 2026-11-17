# Gateway GPT deep analysis of RAD50 (req_main_7c9038d146f6, session ses_9ce133dfbaad4bd0), 2026-11-10

Full text: agent_task_response req_main_7c9038d146f6. Summary:

- Verdict: continue as a bounded, prefill-only, opt-in experiment; not for decode (1 row is slower, 16 rows saves 5.8 us).
- The 768-row win comes from the consumer (GPU read 595 -> 23 us); the CPU producer is ~2.25x slower writing to BAR
  (308 -> 691 us). With RAD42's chunk overlap a slower producer can erase the gain.
- Estimate (inference): 0-3% end-to-end prefill; more under host-read contention; a regression is possible.
- Integration shape: keep RAD42 chunk generations, chunk-leader unconditional poll, bf16 fallback planes, 0x7E01 sentinel,
  summation order; change only the fp16 plane destination and the GPU read pointer. No lab waiter/verifier/ack in serving.
- Open safety questions: DMA_BUF_IOCTL_SYNC is buffer-wide (CPU writing chunk 1 while GPU reads chunk 0 needs testing, or per-chunk
  exports); SFENCE before SYNC_END if streaming stores; slot reuse only after consumption seal; helper-crash quiesce; per-rank planes.
- Next three experiments: (1) per-chunk critical-path trace + A/B with the real mailbox_add_rows_kernel and all three helpers
  (= HL04); (2) BAR store sweep: current 16B storeu / aligned grouping / NT stores + SFENCE / per-thread buffering, at 256/768/2048
  rows, bit-verified; (3) full-layout 8-session contention, reuse and recovery test (gate: repeatable >=2% prefill gain).
- Adaptive policy proposal: pull for <256 rows, push candidate for >=256, chosen at admission and stamped in the request ticket.
