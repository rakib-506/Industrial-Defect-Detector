# Full training pipeline across every category. Run from the project root:
#   .\scripts\train_all.ps1
#   .\scripts\train_all.ps1 -Categories bottle,tile
#   .\scripts\train_all.ps1 -SkipTrained      # reuse existing memory banks
param(
    [string[]]$Categories = @("bottle", "capsule", "carpet", "grid", "metal_nut", "tile"),
    [switch]$SkipTrained,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONWARNINGS = "ignore"

$runArgs = @("-m", "src.run_all", "--categories") + $Categories + @("--nested-cv", "--batch-size", $BatchSize)
if ($SkipTrained) { $runArgs += "--skip-trained" }

Write-Host "`n=== Training: $($Categories -join ', ') ===" -ForegroundColor Cyan
& $py @runArgs
if ($LASTEXITCODE -ne 0) { throw "run_all failed" }

Write-Host "`n=== Comparison table ===" -ForegroundColor Cyan
& $py -m src.compare
if ($LASTEXITCODE -ne 0) { throw "compare failed" }

Write-Host "`n=== Smoke test ===" -ForegroundColor Cyan
& $py -m scripts.smoke_test
if ($LASTEXITCODE -ne 0) { throw "smoke test failed" }

Write-Host "`nDone. Start the API with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000"
