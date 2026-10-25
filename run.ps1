# Run the supplied Django application from the repository root.
$ErrorActionPreference = 'Stop'
$tinkRoot = $PSScriptRoot
$tinkPython = Join-Path $tinkRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $tinkPython)) {
    throw 'Create the environment first with .\setup_windows.ps1, then prepare models (docs/SESSION-SETUP.md).'
}
& $tinkPython (Join-Path $tinkRoot 'Tink-web\start.py') @args
exit $LASTEXITCODE
