$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv-viewer\Scripts\python.exe'
$viewer = Join-Path $PSScriptRoot 'upstream\pico-faces\viewer\view_serial.py'
$outputRoot = Join-Path $PSScriptRoot 'device-output\lesson03-cfg'

$runs = @(
    @{ Name = 'plain'; Cfg = $null }
    @{ Name = 'cfg-4'; Cfg = 4 }
    @{ Name = 'cfg-6'; Cfg = 6 }
    @{ Name = 'cfg-8'; Cfg = 8 }
)

foreach ($run in $runs) {
    $output = Join-Path $outputRoot $run.Name
    Write-Host "Generating seed 42 with $($run.Name)..."
    $arguments = @(
        $viewer
        '--port', 'COM3'
        '--seed', '42'
        '--steps', '4'
        '--class', '1'
        '--out', $output
    )
    if ($null -ne $run.Cfg) {
        $arguments += @('--cfg', [string]$run.Cfg)
    }

    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Generation failed for $($run.Name) with exit code $LASTEXITCODE"
    }
}
