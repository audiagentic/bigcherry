@echo off
rd /s /q H:\development\projects\bigcherry\build\workstation-tune
cmake -B H:\development\projects\bigcherry\build\workstation-tune ^
  -S H:\development\projects\bigcherry\vendor\llama.cpp ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  "-DCMAKE_C_COMPILER=C:/Program Files/AMD/ROCm/7.1/bin/clang.exe" ^
  "-DCMAKE_CXX_COMPILER=C:/Program Files/AMD/ROCm/7.1/bin/clang++.exe" ^
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=ON ^
  -DGGML_HIP=ON -DGGML_HIP_AUTOTUNE=ON -DGGML_HIP_AUTOTUNE_RECORD=ON ^
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=full-max -DGGML_HIP_GRAPHS=ON ^
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_WORKSPACE_METRICS=ON -DGGML_SCHED_MAX_COPIES=4 ^
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_UI=OFF
