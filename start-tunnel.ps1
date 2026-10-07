$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root "venv\Scripts\python.exe"
$Launcher = Join-Path $Root "start-tunnel.py"

if (!(Test-Path -LiteralPath $Python)) {
    throw "Project Python not found: $Python"
}

if (!(Test-Path -LiteralPath $Launcher)) {
    throw "Tunnel launcher not found: $Launcher"
}

& $Python $Launcher @args
exit $LASTEXITCODE
