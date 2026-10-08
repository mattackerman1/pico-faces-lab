$ErrorActionPreference = 'Stop'

$python = Join-Path $PSScriptRoot '.venv-viewer\Scripts\python.exe'
$viewer = Join-Path $PSScriptRoot 'upstream\pico-faces\viewer\view_serial.py'
$outputRoot = Join-Path $PSScriptRoot 'device-output\lesson13-benchmark'
$results = [System.Collections.Generic.List[object]]::new()

$configs = @(
    @{ Mode = 'plain'; Cfg = $null }
    @{ Mode = 'cfg4'; Cfg = 4 }
)

foreach ($config in $configs) {
    foreach ($steps in 1, 2, 4, 8) {
        foreach ($run in 1..3) {
            $output = Join-Path $outputRoot "$($config.Mode)\steps-$steps\run-$run"
            Write-Host "[$($config.Mode)] steps=$steps run=$run"
            $arguments = @(
                $viewer
                '--port', 'COM3'
                '--seed', '42'
                '--steps', [string]$steps
                '--class', '1'
                '--out', $output
            )
            if ($null -ne $config.Cfg) {
                $arguments += @('--cfg', [string]$config.Cfg)
            }

            $lines = & $python @arguments 2>&1
            $text = $lines | Out-String
            Write-Host $text.TrimEnd()
            if ($LASTEXITCODE -ne 0) {
                throw "Generation failed for $($config.Mode), steps=$steps, run=$run"
            }
            if ($text -notmatch '(?<time>\d+) ms, crc (?<crc>[0-9a-f]{8}) \((?<status>[^)]+)\)') {
                throw "Could not parse viewer output for $($config.Mode), steps=$steps, run=$run"
            }

            $results.Add([pscustomobject]@{
                Mode = $config.Mode
                Steps = $steps
                Run = $run
                TimeMs = [int]$Matches.time
                CRC32 = $Matches.crc
                TransferStatus = $Matches.status
            })
        }
    }
}

New-Item -ItemType Directory -Force $outputRoot | Out-Null
$csvPath = Join-Path $outputRoot 'results.csv'
$results | Export-Csv -NoTypeInformation $csvPath

Write-Host "`nSummary"
$results |
    Group-Object Mode, Steps |
    ForEach-Object {
        $times = @($_.Group.TimeMs)
        $crcs = @($_.Group.CRC32 | Sort-Object -Unique)
        [pscustomobject]@{
            Mode = $_.Group[0].Mode
            Steps = $_.Group[0].Steps
            Runs = $_.Count
            MeanMs = [math]::Round(($times | Measure-Object -Average).Average, 1)
            MinMs = ($times | Measure-Object -Minimum).Minimum
            MaxMs = ($times | Measure-Object -Maximum).Maximum
            UniqueCRCs = $crcs.Count
            CRC32 = $crcs -join ','
        }
    } |
    Sort-Object Mode, Steps |
    Format-Table -AutoSize

Write-Host "Results saved to $csvPath"
