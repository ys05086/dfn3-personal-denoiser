$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path 'target/release/pdn_engine.dll')) { throw 'Build the engine first: cargo build --release -p pdn-engine' }
& .venv/Scripts/python.exe -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller install failed' }
# A tool on PATH (for example Poppler) can provide an incompatible icuuc.dll.
# Qt uses Windows' ICU. Restrict dependency discovery to the Python/Windows paths.
$buildOriginalPath = $env:PATH
$basePython = & .venv/Scripts/python.exe -c "import sys; print(sys.base_prefix)"
try {
    $env:PATH = "$PSScriptRoot\.venv\Scripts;$basePython;$env:SystemRoot\System32;$env:SystemRoot"
    & .venv/Scripts/python.exe -m PyInstaller --noconfirm --clean --windowed --onedir --name 'Personal Denoiser' --add-binary 'target/release/pdn_engine.dll;.' --add-data 'app/training;training' --exclude-module torch --exclude-module torchaudio --exclude-module matplotlib --exclude-module pandas --exclude-module IPython --exclude-module tkinter run_app.py
    if ($LASTEXITCODE -ne 0) { throw 'App packaging failed' }
} finally {
    $env:PATH = $buildOriginalPath
}
# Python 3.10 may bundle an older VC runtime than Qt requires. Windows loads the
# root copy first, so ship the matching redistributable from the pinned Qt wheel.
$qtDir = & .venv/Scripts/python.exe -c "import PySide6; from pathlib import Path; print(Path(PySide6.__file__).parent)"
foreach ($runtime in @('vcruntime140.dll', 'vcruntime140_1.dll')) {
    $runtimeSource = Join-Path $qtDir $runtime
    if (Test-Path $runtimeSource) {
        Copy-Item -LiteralPath $runtimeSource -Destination 'dist/Personal Denoiser/_internal' -Force
    }
}
$smoke = Start-Process -FilePath 'dist/Personal Denoiser/Personal Denoiser.exe' -ArgumentList '--smoke-test' -WindowStyle Hidden -PassThru
if (-not $smoke.WaitForExit(30000)) {
    Stop-Process -Id $smoke.Id
    throw 'Packaged app smoke test timed out'
}
if ($smoke.ExitCode -ne 0) { throw "Packaged app smoke test failed ($($smoke.ExitCode))" }
