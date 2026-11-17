# bc_experts: the helpers' expert ownership and .rad container reader, GPU-free (RSC03 slice 2). One source, built by
# the serving component (built and tested there as bc_experts_test; bc-serve starts planning owner maps with it in a later
# slice) and moe-remote (r3_moe_worker selects with it), so the planner and the helper run one implementation. Needs the radiance::abi target (rad_format.h, rad_encoding.h) in the including
# project; moe_backend.h, moe_remote_shm.h and worker_share.h come from components/moe-remote.
if(NOT TARGET bc_experts)
  add_library(bc_experts STATIC "${CMAKE_CURRENT_LIST_DIR}/experts.cpp")
  target_compile_features(bc_experts PUBLIC cxx_std_20)
  target_include_directories(bc_experts PUBLIC "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/../moe-remote")
  # The guarded expert-layout hook creates v2::ConfigError values in experts.cpp.
  # This is a library dependency, not an optional obligation of each consumer.
  if(NOT TARGET bc_serving_v2)
    message(FATAL_ERROR "bc_experts: include bc_serving_v2.cmake first")
  endif()
  target_link_libraries(bc_experts PUBLIC radiance::abi bc_serving_v2)
endif()
