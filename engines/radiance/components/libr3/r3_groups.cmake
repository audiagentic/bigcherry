# Build-only grouping for libr3's upstream translation units.
#
# These categories never alter the runtime plugin, row registry, or compatibility
# shims. Every unit has one compilation owner; model profiles select groups for
# development and tests, not separate copies of the kernels.
function(r3_kernel_group unit out)
  if(unit MATCHES "^r4d_ar_")
    set(group collectives)
  elseif(unit MATCHES "^r4d_(dflash_|ds41_walk$)")
    set(group speculation)
  elseif(unit MATCHES "^r4d_moe" OR unit MATCHES "^r4d_(expert_|situ$)")
    set(group moe)
  elseif(unit MATCHES "^r4d_(attn_|sam_attn_|qsa_|kv_|qkv_|qk_|dsa$|mla$|kda$|ssd$|ds41_sparse$)")
    set(group attention)
  elseif(unit MATCHES "^r4d_(gen$|audio$|video$|ltx$|upscaler$|vsa$|vit_|encoder_|sam_|lili$|posemb$)")
    set(group multimodal)
  elseif(unit MATCHES "^r4d_gemm_(bf16|f32)" OR unit MATCHES "^r4d_lora_")
    set(group gemm_dense)
  elseif(unit MATCHES "^r4d_gemm_" OR unit MATCHES "^r4d_(iqr_embed|dequant_w|gram_accum|logits_gemm_i8$|ds41_ggml|ggml_wmma$)")
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

function(r3_read_unit_lines upstream out)
  # The unit list and each unit's flags are libr4d's own, read from its build file: `"unit|flags"`.
  file(STRINGS "${upstream}/libr4d/CMakeLists.txt" unit_lines REGEX "^  \"r4d_[A-Za-z0-9_]+\\|")
  # Unit names have capitals (r4d_ar_oneshot_Nrank_exact): a lowercase-only pattern once dropped the 3/4/8-rank
  # all-reduce units without a word, and their entry points became stubs. Every unit line must parse.
  file(STRINGS "${upstream}/libr4d/CMakeLists.txt" all_unit_lines REGEX "^  \"r4d_")
  list(LENGTH unit_lines n_parsed)
  list(LENGTH all_unit_lines n_listed)
  if(NOT n_parsed EQUAL n_listed OR n_parsed EQUAL 0)
    message(FATAL_ERROR "libr3: parsed ${n_parsed} of libr4d's ${n_listed} unit lines; fix the unit pattern")
  endif()
  set(${out} "${unit_lines}" PARENT_SCOPE)
endfunction()

function(r3_unit_fields line unit_out flags_out)
  string(REGEX MATCH "\"(r4d_[A-Za-z0-9_]+)\\|([^\"]*)\"" _ "${line}")
  set(unit ${CMAKE_MATCH_1})
  set(flags "${CMAKE_MATCH_2}")
  if(NOT unit)
    message(FATAL_ERROR "libr3: invalid unit line: ${line}")
  endif()
  set(${unit_out} "${unit}" PARENT_SCOPE)
  set(${flags_out} "${flags}" PARENT_SCOPE)
endfunction()

# Read-only canonical classification of every upstream unit, including units
# excluded from a particular component build. The build's r3_groups.tsv still
# describes only the actual configured objects. Use the same TSV format and
# classifier; do not create a second taxonomy.
if(DEFINED CMAKE_SCRIPT_MODE_FILE AND CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  cmake_minimum_required(VERSION 3.21)
  if(NOT DEFINED R3_GROUP_SOURCE OR NOT DEFINED R3_GROUP_OUTPUT)
    message(FATAL_ERROR "Set R3_GROUP_SOURCE and R3_GROUP_OUTPUT for the inventory")
  endif()
  r3_read_unit_lines("${R3_GROUP_SOURCE}" lines)
  set(seen "")
  set(manifest "group\tunit\n")
  foreach(line IN LISTS lines)
    r3_unit_fields("${line}" unit flags)
    if(unit IN_LIST seen)
      message(FATAL_ERROR "Duplicate upstream unit: ${unit}")
    endif()
    list(APPEND seen "${unit}")
    r3_kernel_group("${unit}" group)
    if(NOT group IN_LIST R3_KERNEL_GROUPS)
      message(FATAL_ERROR "Unknown group: ${group}")
    endif()
    string(APPEND manifest "${group}\t${unit}\n")
  endforeach()
  file(WRITE "${R3_GROUP_OUTPUT}" "${manifest}")
endif()
