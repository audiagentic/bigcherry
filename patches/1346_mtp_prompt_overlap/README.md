# 1346_mtp_prompt_overlap

QFP31 implementation package. Chunk 1 contains measurement only.

Set `BIGCHERRY_MTP_PROMPT_TIMING=1` for a single-slot prompt. At the prompt-to-generation boundary it writes:

`BIGCHERRY_MTP_PROMPT_TIMING target_nextn_ms=<...> draft_process_ms=<...> draft_decode_ms=<...> chunks=<...> tokens=<...>`

`draft_process_ms` is inclusive MTP `process()` wall time. `draft_decode_ms` is the wall time of the existing draft-context prompt catch-up `llama_process()` calls. `target_nextn_ms` is host-visible time in target NextN buffer acquisition plus per-row NextN getters/copies; it may include synchronization forced by those getters. The residual `draft_process_ms - target_nextn_ms - draft_decode_ms` is other synchronous host work in the MTP process hook.

The diagnostic intentionally attributes only batches containing one sequence. Use the requested single-slot Gate-0 run for the cost split.
