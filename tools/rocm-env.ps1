<# Compatibility wrapper; dot-source the canonical environment helper. #>
. (Join-Path $PSScriptRoot 'env\rocm-env.ps1') @args
