# rad_add_plugin gives each device source a -cuid hashed from its path relative to the component's source directory.
# For a unit wrapper generated in the build tree that path holds the build directory (the build plan's id), so every
# new plan changed every unit's command line and every unit missed the compiler cache. r3_build_tree_cuid replaces
# the ID of each such source with one hashed from its path inside the build tree: still unique per source, and the
# same wherever the build tree lives. Call it after rad_add_plugin.
function(r3_build_tree_cuid plugin)
  foreach(src ${ARGN})
    get_source_file_property(opts ${src} COMPILE_OPTIONS)
    if(NOT opts MATCHES "-cuid=")
      message(FATAL_ERROR "${plugin}: rad_add_plugin no longer sets -cuid on ${src}; revisit r3_cuid.cmake")
    endif()
    file(RELATIVE_PATH rel ${CMAKE_CURRENT_BINARY_DIR} ${src})
    string(SHA256 cuid "${plugin}/${rel}")
    string(SUBSTRING "${cuid}" 0 16 cuid)
    list(TRANSFORM opts REPLACE "^-cuid=.*" "-cuid=${cuid}")
    set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "${opts}")
  endforeach()
endfunction()
