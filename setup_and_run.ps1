# One-click setup and benchmark for Scout on Windows. Started by setup_and_run.bat.
# Your API key is typed into a hidden prompt, kept only in this window's memory, and never written to disk.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force results | Out-Null
Start-Transcript -Path (Join-Path $PSScriptRoot "results\setup-log.txt") -Force | Out-Null

function Step($text) { Write-Host ""; Write-Host "=== $text ===" -ForegroundColor Cyan }
function Fail($text) {
    Write-Host ""; Write-Host "STOPPED: $text" -ForegroundColor Red
    Write-Host "Tell Claude it stopped. The log is in results\setup-log.txt."
    Stop-Transcript | Out-Null
    exit 1
}

Step "1/8 Python environment"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command py -ErrorAction SilentlyContinue }
if (-not $python) { Fail "Python was not found. Install Python 3 from python.org or the Microsoft Store, then run this again." }
if (-not (Test-Path ".venv")) { & $python.Source -m venv .venv; if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment." } }
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install -q --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Installing the Python packages failed." }

Step "2/8 Download the FastAPI docs"
& $py scripts\fetch_corpus.py
if ($LASTEXITCODE -ne 0) { Fail "Downloading the docs failed." }

Step "3/8 Run the tests"
& $py -m unittest discover -s tests -t .
if ($LASTEXITCODE -ne 0) { Fail "Some tests failed. See the output above." }

Step "4/8 Retrieval evaluation (free, no API key)"
& $py evals\retrieval_eval.py
if ($LASTEXITCODE -ne 0) { Fail "The retrieval evaluation failed." }

Step "5/8 Ollama (local models)"
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
if (-not $ollama) {
    Write-Host "Ollama is not installed. Installing it with winget (a permission prompt may appear)..."
    try { winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements } catch { }
    $ollama = Find-Ollama
}
if (-not $ollama) {
    Write-Host "Could not install Ollama automatically. Continuing with Claude models only." -ForegroundColor Yellow
    $useOllama = $false
} else {
    $up = $false
    for ($i = 0; $i -lt 15 -and -not $up; $i++) {
        try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 2 | Out-Null; $up = $true } catch {
            if ($i -eq 0) { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden }
            Start-Sleep -Seconds 2
        }
    }
    if (-not $up) { Write-Host "Ollama did not start. Continuing with Claude models only." -ForegroundColor Yellow; $useOllama = $false }
    else {
        Write-Host "Downloading the llama3.1 model (about 5 GB, only the first time)..."
        & $ollama pull llama3.1
        if ($LASTEXITCODE -ne 0) { Write-Host "Model download failed. Continuing with Claude models only." -ForegroundColor Yellow; $useOllama = $false }
    }
}

Step "6/8 Your Anthropic API key"
Write-Host "Paste your API key and press Enter. Nothing will show on screen as you paste, that is normal."
$secure = Read-Host -AsSecureString "API key"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Clear-Host
if (-not $env:ANTHROPIC_API_KEY) { Fail "No key was entered." }
Write-Host "Key received (hidden)." -ForegroundColor Green

& $py scripts\check_key.py
if ($LASTEXITCODE -ne 0) { Fail "Your API key was not accepted. Run this again and paste the key carefully (right-click to paste, then Enter)." }

Step "7/8 Quick check with 2 questions (costs a few cents)"
& $py evals\run_eval.py --models anthropic:claude-haiku-5-5 --judge anthropic:claude-opus-5-5 --limit 2 --max-cost 0.5 --out results\smoke
if ($LASTEXITCODE -ne 0) { Fail "The quick check failed. This usually means a wrong key or a model name the API does not accept." }

Step "8/8 Full benchmark (spend capped at 3 dollars including the judge)"
$models = @("anthropic:claude-haiku-5-5", "anthropic:claude-sonnet-5-5")
if ($useOllama) { $models += "ollama:llama3.1" }
& $py evals\run_eval.py --models $models --judge anthropic:claude-opus-5-5 --max-cost 3 --out results
if ($LASTEXITCODE -ne 0) { Fail "The benchmark failed. Partial output may be in the results folder." }

Step "Done: starting the app"
Start-Process $py -ArgumentList "run.py" -WorkingDirectory $PSScriptRoot
Start-Sleep -Seconds 4
Start-Process "http://localhost:8000"
Write-Host ""
Write-Host "Benchmark finished and the app is running at http://localhost:8000" -ForegroundColor Green
Write-Host "Go back to Claude and say done. Leave this window open so the app keeps its key."
Stop-Transcript | Out-Null
