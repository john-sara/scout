# Second pass: fix-and-rerun. Started by rerun.bat. Your API key is typed into a hidden prompt and never saved.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force results | Out-Null
Start-Transcript -Path (Join-Path $PSScriptRoot "results\rerun-log.txt") -Force | Out-Null
function Step($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }
function Fail($text) {
    Write-Host ""; Write-Host "STOPPED: $text" -ForegroundColor Red
    Write-Host "Tell Claude it stopped. The log is in results\rerun-log.txt."
    Stop-Transcript | Out-Null
    exit 1
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Fail "The Python environment is missing. Run setup_and_run.bat first." }
$v1 = Get-ChildItem results\run-*.json | Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $v1) { Fail "No earlier benchmark run was found in the results folder." }

Step "1/6 Run the tests"
& $py -m unittest discover -s tests -t .
if ($LASTEXITCODE -ne 0) { Fail "Some tests failed." }

Step "2/6 Ollama (local models)"
$useOllama = $true
function Find-Ollama {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $cmd = Get-Command ollama -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $guess) { return $guess }
    return $null
}
$ollama = Find-Ollama
if (-not $ollama) { Write-Host "Ollama is not installed, skipping local models." -ForegroundColor Yellow; $useOllama = $false }
else {
    $up = $false
    for ($i = 0; $i -lt 20 -and -not $up; $i++) {
        try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; $up = $true } catch {
            if ($i -eq 0) { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden }
            Start-Sleep -Seconds 3
        }
    }
    if (-not $up) { Write-Host "The Ollama server did not start, skipping local models." -ForegroundColor Yellow; $useOllama = $false }
    else {
        Write-Host "Making sure the llama3.1 model is downloaded (about 5 GB the first time)..."
        & $ollama pull llama3.1
        if ($LASTEXITCODE -ne 0) { Write-Host "Model download failed, skipping local models." -ForegroundColor Yellow; $useOllama = $false }
    }
}

Step "3/6 Your Anthropic API key"
Write-Host "Paste your NEW API key and press Enter. Nothing will show as you paste, that is normal."
$secure = Read-Host -AsSecureString "API key"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Clear-Host
if (-not $env:ANTHROPIC_API_KEY) { Fail "No key was entered." }
Write-Host "Key received (hidden)." -ForegroundColor Green

Step "4/6 Re-grade the first run with the fixed judge (cap 1 dollar)"
& $py evals\regrade.py $v1.FullName --judge anthropic:claude-opus-5-5 --max-cost 1.0 --out results
if ($LASTEXITCODE -ne 0) { Fail "Re-grading failed." }

Step "5/6 Second benchmark with the agent fixes (cap 2 dollars)"
& $py evals\run_eval.py --models anthropic:claude-haiku-5-5 anthropic:claude-sonnet-5-5 --judge anthropic:claude-opus-5-5 --max-cost 2.0 --out results\v2
if ($LASTEXITCODE -ne 0) { Fail "The second benchmark failed. Partial output may be in results\v2." }

if ($useOllama) {
    Step "6/6 Local model, llama3.1 (free to run, judge cap 0.60 dollars)"
    & $py evals\run_eval.py --models ollama:llama3.1 --judge anthropic:claude-opus-5-5 --max-cost 0.6 --out results\ollama
    if ($LASTEXITCODE -ne 0) { Write-Host "The local-model run failed. The Claude results are still saved." -ForegroundColor Yellow }
} else { Step "6/6 Local model skipped" }

Step "Building the results page"
& $py evals\report.py
if ($LASTEXITCODE -ne 0) { Write-Host "The results page could not be built. Tell Claude." -ForegroundColor Yellow }

Step "Restarting the app with the fixes"
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like "*run.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Process $py -ArgumentList "run.py" -WorkingDirectory $PSScriptRoot
Start-Sleep -Seconds 4
Start-Process "http://localhost:8000"
Start-Process (Join-Path $PSScriptRoot "results\report.html")
Write-Host ""
Write-Host "All done. Go back to Claude and say done. Leave this window open so the app keeps its key." -ForegroundColor Green
Stop-Transcript | Out-Null
