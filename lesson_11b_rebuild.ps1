$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv-fold\Scripts\python.exe'
$repo = Join-Path $PSScriptRoot 'upstream\pico-faces'

$zigCommand = Get-Command zig -ErrorAction SilentlyContinue
if ($zigCommand) {
    $zig = $zigCommand.Source
} else {
    $packageRoot = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    $package = Get-ChildItem $packageRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object Name -Like 'zig.zig_*' |
        Select-Object -First 1
    $zig = Get-ChildItem $package.FullName -Filter 'zig.exe' -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $zig) {
    throw 'Zig was not found.'
}

Push-Location $repo
try {
    $latentDirectory = 'artifacts\m3_long_cfg\runs\vae'
    New-Item -ItemType Directory -Force $latentDirectory | Out-Null
    $latentDestination = Join-Path $latentDirectory 'latents.npz'
    if (-not (Test-Path $latentDestination)) {
        Copy-Item 'checkpoints\m3_long_cfg\latent_stats.npz' $latentDestination
    }

    Write-Host 'Folding and exporting the released fast model...'
    & $python 'quant\fold.py' --model m3_long_cfg
    if ($LASTEXITCODE -ne 0) {
        throw "Fold/export failed with exit code $LASTEXITCODE"
    }

    $rebuiltBlob = Resolve-Path 'artifacts\m3_long_cfg\export\model.bin'
    $releasedBlob = Resolve-Path 'checkpoints\m3_long_cfg\model.bin'
    $rebuiltBytes = [System.IO.File]::ReadAllBytes($rebuiltBlob)
    $releasedBytes = [System.IO.File]::ReadAllBytes($releasedBlob)
    $blobMatches = [System.Linq.Enumerable]::SequenceEqual($rebuiltBytes, $releasedBytes)

    Write-Host "rebuilt blob bytes:  $($rebuiltBytes.Length)"
    Write-Host "released blob bytes: $($releasedBytes.Length)"
    Write-Host "blob byte-identical:  $blobMatches"
    Get-FileHash $rebuiltBlob -Algorithm SHA256
    Get-FileHash $releasedBlob -Algorithm SHA256
    if (-not $blobMatches) {
        Write-Warning 'The rebuilt model.bin differs from the released reference; continuing with internal C-vs-simulator verification.'
    }

    Write-Host 'Compiling the C engine against the rebuilt rf_cfg.h...'
    New-Item -ItemType Directory -Force 'build\m3_long_cfg-rebuilt\e2e' | Out-Null
    $sources = @(
        'engine/desktop/main_golden.c'
        'engine/src/graph.c'
        'engine/src/dit.c'
        'engine/src/vae_dec.c'
        'engine/src/kernels_ref.c'
        'engine/src/prng.c'
    )
    $compileArguments = @(
        'cc'
        '-O2'
        '-Wall'
        '-Wextra'
        '-Iartifacts/m3_long_cfg/export'
        '-Iengine/include'
    ) + $sources + @('-o', 'build/m3_long_cfg-rebuilt/rf_golden.exe')
    & $zig @compileArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Compilation failed with exit code $LASTEXITCODE"
    }

    $goldenDirectory = 'artifacts\m3_long_cfg\goldens\e2e_trained'
    $seedFiles = Get-ChildItem $goldenDirectory -Filter 'golden_*.rgb' | Sort-Object Name
    $seeds = @($seedFiles | ForEach-Object { $_.BaseName -replace '^golden_', '' })
    & '.\build\m3_long_cfg-rebuilt\rf_golden.exe' `
        $rebuiltBlob `
        'build\m3_long_cfg-rebuilt\e2e' `
        4 `
        @seeds
    if ($LASTEXITCODE -ne 0) {
        throw "C generation failed with exit code $LASTEXITCODE"
    }

    foreach ($seed in $seeds) {
        $generated = [System.IO.File]::ReadAllBytes(
            (Resolve-Path "build\m3_long_cfg-rebuilt\e2e\eng_$seed.rgb"))
        $golden = [System.IO.File]::ReadAllBytes(
            (Resolve-Path "$goldenDirectory\golden_$seed.rgb"))
        $matches = [System.Linq.Enumerable]::SequenceEqual($generated, $golden)
        Write-Host "seed ${seed}: byte-exact = $matches"
        if (-not $matches) {
            throw "C output for seed $seed differs from the rebuilt golden."
        }
    }
} finally {
    Pop-Location
}
