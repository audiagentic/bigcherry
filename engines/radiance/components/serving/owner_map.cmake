# bc_owner_map: the manifest's owner map checked against the GPU-free planner (owner_map.h; RSC08 slice S1). Built from
# the serving component; needs the bc_serving_v1 and bc_experts targets from the including project (the helper links it
# at S4a to verify its own owned set).
if(NOT TARGET bc_owner_map)
  add_library(bc_owner_map STATIC "${CMAKE_CURRENT_LIST_DIR}/owner_map.cpp")
  target_link_libraries(bc_owner_map PUBLIC bc_serving_v1 bc_experts)
  target_include_directories(bc_owner_map PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
endif()
