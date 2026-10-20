# Re-grades the second benchmark with the fixed judge. Started by regrade_final.bat. Your API key is typed into a hidden prompt and never saved.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
Start-Transcript -Path (Join-Path $PSScriptRoot "results\regrade-final-log.txt") -Force | Out-Null
function Fail($text) {
    Write-Host ""; Write-Host "STOPPED: $text" -ForegroundColor Red
    Write-Host "Tell Claude it stopped. The log is in results\regrade-final-log.txt."
    Stop-Transcript | Out-Null
    exit 1
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Fail "The Python environment is missing." }
$v2 = Get-ChildItem results\v2\run-*.json | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $v2) { Fail "No second benchmark run was found in results\v2." }

Write-Host "Paste your NEW API key and press Enter. Nothing will show as you paste, that is normal."
$secure = Read-Host -AsSecureString "API key"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Clear-Host
if (-not $env:ANTHROPIC_API_KEY) { Fail "No key was entered." }
Write-Host "Key received (hidden)." -ForegroundColor Green
& $py scripts\check_key.py
if ($LASTEXITCODE -ne 0) { Fail "Your API key was not accepted. Run this again and paste the key carefully." }

Write-Host "Re-grading (about 1 dollar, capped at 2)..." -ForegroundColor Cyan
& $py evals\regrade.py $v2.FullName --judge anthropic:claude-opus-5-5 --max-cost 2.0 --out results\v2 --write-run
if ($LASTEXITCODE -ne 0) { Fail "Re-grading failed." }

Write-Host "Building the results page..." -ForegroundColor Cyan
& $py evals\report.py
Write-Host ""; Write-Host "All done. Go back to Claude and say done. If the app window is open, close it and run rerun.bat's app step, or just tell me." -ForegroundColor Green
Stop-Transcript | Out-Null
