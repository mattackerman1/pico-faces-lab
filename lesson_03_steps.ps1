$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv-viewer\Scripts\python.exe'
$viewer = Join-Path $PSScriptRoot 'upstream\pico-faces\viewer\view_serial.py'
$outputRoot = Join-Path $PSScriptRoot 'device-output\lesson03-steps'

foreach ($steps in 1, 2, 4, 8) {
    $output = Join-Path $outputRoot "steps-$steps"
    Write-Host "Generating seed 42 with $steps step(s)..."
    & $python $viewer `
        --port COM3 `
        --seed 42 `
        --steps $steps `
        --class 1 `
        --cfg 4 `
        --out $output

    if ($LASTEXITCODE -ne 0) {
        throw "Generation failed at $steps step(s) with exit code $LASTEXITCODE"
    }
}
