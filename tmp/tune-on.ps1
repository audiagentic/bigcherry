# Slice A runtime proof - ON run (gfx1100, 7900 GRE)
$ErrorActionPreference = "Stop"
$env:PATH = "C:\Program Files\AMD\ROCm\7.1\bin;C:\Program Files\AMD\ROCm\7.1\lib;" + $env:PATH
$env:HIP_VISIBLE_DEVICES = "0"
$env:GGML_HIP_DISPATCH_MODE = "tune"
$env:GGML_HIP_TUNE_SCREEN_SAMPLES = "3"
$env:GGML_HIP_TUNE_FINAL_SAMPLES = "5"
$env:GGML_HIP_DISPATCH_DB = "H:\development\projects\bigcherry\tmp\tune-on.jsonl"
$env:GGML_CUDA_DISABLE_GRAPHS = "1"
Remove-Item Env:GGML_HIP_TUNE_DOUBLE_NATIVE -ErrorAction SilentlyContinue  # default ON
& "H:\development\projects\bigcherry\build\workstation-tune\bin\test-backend-ops.exe" test -o MUL_MAT
