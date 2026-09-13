param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& $Python -m streamlit run (Join-Path $PSScriptRoot "app.py")
