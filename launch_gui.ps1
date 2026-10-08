$ErrorActionPreference = 'Stop'

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv-viewer\Scripts\python.exe'
$app = Join-Path $projectRoot 'pico_faces_gui.py'

if (-not (Test-Path $python)) {
    throw "Viewer environment not found at $python"
}

& $python $app --port COM3
