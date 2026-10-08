$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv-viewer\Scripts\python.exe'
$viewer = Join-Path $PSScriptRoot 'upstream\pico-faces\viewer\view_serial.py'
$outputRoot = Join-Path $PSScriptRoot 'device-output\lesson03-classes'

foreach ($class in 0, 1, 2, 3, 4) {
    $output = Join-Path $outputRoot "class-$class"
    Write-Host "Generating seed 42 with class $class..."
    & $python $viewer `
        --port COM3 `
        --seed 42 `
        --steps 4 `
        --class $class `
        --cfg 4 `
        --out $output

    if ($LASTEXITCODE -ne 0) {
        throw "Generation failed for class $class with exit code $LASTEXITCODE"
    }
}
