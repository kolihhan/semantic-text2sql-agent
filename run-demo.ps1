$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $Root '.run'
$TempDir = Join-Path $RunDir 'tmp'
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null
$env:TEMP = $TempDir
$env:TMP = $TempDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/ and re-run.'
}

Push-Location $Root
try {
    uv sync
    uv run semantic-sql demo
} finally {
    Pop-Location
}
