# scripts/build_exe.ps1
# Build WDS_Optimizer.exe via PyInstaller (onedir mode).
# Pure ASCII, single quotes only. Should work on PS 5.1 and PS 7+.

$ErrorActionPreference = 'Stop'

Write-Host '=== WDS Optimizer - Build ===' -ForegroundColor Cyan

# 1. Check venv
if (-not $env:VIRTUAL_ENV) {
    Write-Host 'No active venv. Activate .venv first:' -ForegroundColor Red
    Write-Host '  .\.venv\Scripts\Activate.ps1' -ForegroundColor Yellow
    exit 1
}
Write-Host "Venv: $env:VIRTUAL_ENV" -ForegroundColor Gray

# 2. Ensure PyInstaller is importable
$needInstall = $false
try {
    python -c 'import PyInstaller' 2>$null
    if ($LASTEXITCODE -ne 0) { $needInstall = $true }
} catch {
    $needInstall = $true
}

if ($needInstall) {
    Write-Host 'Installing PyInstaller...' -ForegroundColor Yellow
    python -m pip install 'pyinstaller>=6.0'
} else {
    Write-Host 'PyInstaller already installed.' -ForegroundColor Gray
}

# 3. Clean
Write-Host 'Cleaning build/ dist/ ...' -ForegroundColor Yellow
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# 4. Build
Write-Host 'Running PyInstaller ...' -ForegroundColor Yellow
$start = Get-Date
python -m PyInstaller WDS_Optimizer.spec --clean --noconfirm

if ($LASTEXITCODE -ne 0) {
    Write-Host 'Build FAILED' -ForegroundColor Red
    exit 1
}

$elapsed = (Get-Date) - $start
Write-Host "Build finished in $([int]$elapsed.TotalSeconds) s" -ForegroundColor Green

# 5. Sanity check
$exe = 'dist\WDS_Optimizer\WDS_Optimizer.exe'
if (Test-Path $exe) {
    $sizeBytes = (Get-ChildItem 'dist\WDS_Optimizer' -Recurse |
                  Measure-Object Length -Sum).Sum
    $sizeMB = [math]::Round($sizeBytes / 1MB, 1)
    Write-Host "OK: $exe" -ForegroundColor Green
    Write-Host "Total size: $sizeMB MB" -ForegroundColor Green
} else {
    Write-Host 'Build reported OK but exe missing' -ForegroundColor Red
    exit 1
}

# 6. Final instructions (no variable interpolation, no quotes)
Write-Host ''
Write-Host 'To test:' -ForegroundColor Cyan
Write-Host '  .\dist\WDS_Optimizer\WDS_Optimizer.exe' -ForegroundColor Yellow
Write-Host ''
Write-Host 'To package as ZIP for release:' -ForegroundColor Cyan
Write-Host '  Compress-Archive -Path dist\WDS_Optimizer -DestinationPath WDS_Optimizer_v0.6.0_windows_x64.zip' -ForegroundColor Yellow