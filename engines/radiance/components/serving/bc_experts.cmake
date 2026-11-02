# bc_experts: the helpers' expert ownership and .rad container reader, GPU-free (RSC03 slice 2). One source, built by
# both the serving component (bc-serve plans owner maps with it) and moe-remote (r3_moe_worker selects with it), so
# the two run one implementation. Needs the radiance::abi target (rad_format.h, rad_encoding.h) in the including
# project; moe_backend.h, moe_remote_shm.h and worker_share.h come from components/moe-remote.
if(NOT TARGET bc_experts)
  add_library(bc_experts STATIC "${CMAKE_CURRENT_LIST_DIR}/experts.cpp")
  target_compile_features(bc_experts PUBLIC cxx_std_20)
  target_include_directories(bc_experts PUBLIC "${CMAKE_CURRENT_LIST_DIR}" "${CMAKE_CURRENT_LIST_DIR}/../moe-remote")
  target_link_libraries(bc_experts PUBLIC radiance::abi)
endif()
