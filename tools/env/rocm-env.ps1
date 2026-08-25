<# Select a vendored ROCm/HIP toolchain version. Dot-source this script. #>
param(
    [Parameter(Position = 0)]
    [string]$Version,
    [switch]$List
)

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RocmVendorRoot = Join-Path $RepoRoot 'vendor\rocm'

function Get-VendoredRocmVersions {
    if (-not (Test-Path $RocmVendorRoot)) { return @() }
    Get-ChildItem -Path $RocmVendorRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\hipcc.exe') } |
        Select-Object -ExpandProperty Name |
        Sort-Object
}

if ($List -or -not $Version) {
    $versions = Get-VendoredRocmVersions
    if (-not $versions) {
        Write-Host "No vendored ROCm installs found under $RocmVendorRoot"
        Write-Host "Populate one with: robocopy 'C:\Program Files\AMD\ROCm\<ver>' '$RocmVendorRoot\<ver>' /E"
    } else {
        Write-Host "Vendored ROCm versions under $RocmVendorRoot :"
        $versions | ForEach-Object {
            $mark = if ($env:ROCM_PATH -eq (Join-Path $RocmVendorRoot $_)) { ' (active)' } else { '' }
            Write-Host "  $_$mark"
        }
    }
    return
}

$Target = Join-Path $RocmVendorRoot $Version
$HipccPath = Join-Path $Target 'bin\hipcc.exe'
if (-not (Test-Path $HipccPath)) {
    Write-Error "No vendored ROCm '$Version' at $Target (expected $HipccPath). Run '. tools/env/rocm-env.ps1 -List' to see what's available."
    return
}

$env:ROCM_PATH = $Target
$env:HIP_PATH = $Target
$BinPath = Join-Path $Target 'bin'
$env:PATH = "$BinPath;" + ($env:PATH -split ';' | Where-Object { $_ -notlike "$RocmVendorRoot*" }) -join ';'

Write-Host "ROCm $Version selected: $Target"
& hipcc --version 2>&1 | Select-Object -First 1
