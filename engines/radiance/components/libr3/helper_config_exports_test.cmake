# Inspect symbols without loading HIP or initializing a device.
execute_process(COMMAND "${NM}" -D "${LIBRARY}"
  RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect kernel configuration ABI: ${error}")
endif()
foreach(symbol bc_r4d_configure_helper bc_r4d_helper_config)
  if(NOT symbols MATCHES "[0-9a-fA-F]+ T ${symbol}[\r\n]")
    message(FATAL_ERROR "Kernel library does not export ${symbol}")
  endif()
endforeach()
