# Build-only grouping for libr3's upstream translation units.
#
# These categories never alter the runtime plugin, row registry, or compatibility
# shims. Every unit has one compilation owner; model profiles select groups for
# development and tests, not separate copies of the kernels.
function(r3_kernel_group unit out)
  if(unit MATCHES "^r4d_ar_")
    set(group collectives)
  elseif(unit MATCHES "^r4d_dflash_")
    set(group speculation)
  elseif(unit MATCHES "^r4d_moe" OR unit MATCHES "^r4d_expert_")
    set(group moe)
  elseif(unit MATCHES "^r4d_(attn_|sam_attn_|qsa_|kv_|qkv_|qk_|dsa$|mla$|kda$|ssd$)")
    set(group attention)
  elseif(unit MATCHES "^r4d_(gen$|audio$|video$|ltx$|upscaler$|vsa$|vit_|encoder_|sam_|lili$)")
    set(group multimodal)
  elseif(unit MATCHES "^r4d_gemm_(bf16|f32)" OR unit MATCHES "^r4d_lora_")
    set(group gemm_dense)
  elseif(unit MATCHES "^r4d_gemm_" OR unit MATCHES "^r4d_(iqr_embed|dequant_w|gram_accum)")
    set(group gemm_quant)
  else()
    # Shared small operators (norm, quant, sampling, state, layout, etc.).
    # Unknown new upstream units remain buildable and visible in core until
    # intentionally assigned to a specialist family.
    set(group core)
  endif()
  set(${out} "${group}" PARENT_SCOPE)
endfunction()

set(R3_KERNEL_GROUPS
  core gemm_quant gemm_dense attention moe speculation multimodal collectives)
