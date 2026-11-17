# Fail the component test suite if a new injected callable can bypass Callback.
# Internal recursion helpers outside session are not external owner callbacks.
foreach(name IN ITEMS session.h session.cpp)
  file(READ "${CMAKE_CURRENT_LIST_DIR}/${name}" source)
  if(source MATCHES "std::function[ \t\r\n]*<|std::move_only_function[ \t\r\n]*<|function_ref[ \t\r\n]*<")
    message(FATAL_ERROR "${name}: raw callable storage outside callback_boundary.h")
  endif()
  if(source MATCHES "std::invoke[ \t\r\n]*\\(")
    message(FATAL_ERROR "${name}: unguarded std::invoke bypass")
  endif()
endforeach()
