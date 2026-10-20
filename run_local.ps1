# Runs only the local-model benchmark (llama3.1 through Ollama), then rebuilds the results page.
# Your API key is typed into a hidden prompt (the Claude judge needs it) and never saved.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
Start-Transcript -Path (Join-Path $PSScriptRoot "results\local-log.txt") -Force | Out-Null
function Fail($text) {
    Write-Host ""; Write-Host "STOPPED: $text" -ForegroundColor Red
    Write-Host "Tell Claude it stopped. The log is in results\local-log.txt."
    Stop-Transcript | Out-Null
    exit 1
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Fail "The Python environment is missing. Run setup_and_run.bat first." }

$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue)
if ($ollama) { $ollama = $ollama.Source } else {
    $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $guess) { $ollama = $guess } else { Fail "Ollama is not installed." }
}
$up = $false
for ($i = 0; $i -lt 20 -and -not $up; $i++) {
    try { Invoke-RestMethod http://localhost:11434/api/tags -TimeoutSec 3 | Out-Null; $up = $true } catch {
        if ($i -eq 0) { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden }
        Start-Sleep -Seconds 3
    }
}
if (-not $up) { Fail "The Ollama server did not start." }
$tags = Invoke-RestMethod http://localhost:11434/api/tags
if (-not ($tags.models | Where-Object { $_.name -like "llama3.1*" })) { Fail "llama3.1 is not downloaded yet. Run: ollama pull llama3.1" }
Write-Host "Ollama is running with llama3.1." -ForegroundColor Green

Write-Host ""; Write-Host "Paste your API key and press Enter. Nothing will show as you paste, that is normal."
$secure = Read-Host -AsSecureString "API key"
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$env:ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
Clear-Host
if (-not $env:ANTHROPIC_API_KEY) { Fail "No key was entered." }

Write-Host "Running llama3.1 on all 33 questions. This is free to run, but slow on a local GPU." -ForegroundColor Cyan
& $py evals\run_eval.py --models ollama:llama3.1 --judge anthropic:claude-opus-5-5 --max-cost 0.6 --out results\ollama
if ($LASTEXITCODE -ne 0) { Fail "The local-model run failed." }
& $py evals\report.py
Start-Process (Join-Path $PSScriptRoot "results\report.html")
Write-Host ""; Write-Host "Done. Go back to Claude and say done." -ForegroundColor Green
Stop-Transcript | Out-Null
