# Equivalente de run.sh para Windows PowerShell.
#   .\run.ps1          → build + up + espera a que la API esté sana
#   .\run.ps1 down     → detiene los contenedores
#   .\run.ps1 logs     → sigue los logs de la API
param([string]$Command = "up")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Command -eq "down") { docker compose down; exit 0 }
if ($Command -eq "logs") { docker compose logs -f api; exit 0 }

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "-> Se creo .env a partir de .env.example (editalo para usar Anthropic/OpenAI/Pinecone/Tavily)."
}

$envVars = @{}
Get-Content .env | Where-Object { $_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$' } | ForEach-Object {
    $envVars[$Matches[1]] = $Matches[2]
}
$apiPort = if ($envVars["API_PORT"]) { $envVars["API_PORT"] } else { "8000" }
$phoenixPort = if ($envVars["PHOENIX_PORT"]) { $envVars["PHOENIX_PORT"] } else { "6006" }

docker compose up --build -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$apiUrl = "http://localhost:$apiPort"
Write-Host -NoNewline "-> Esperando a la API en $apiUrl "
for ($i = 0; $i -lt 90; $i++) {
    try {
        $health = Invoke-RestMethod "$apiUrl/health" -TimeoutSec 3
        Write-Host " OK"
        $health | ConvertTo-Json
        Write-Host ""
        Write-Host "  API (Swagger) : $apiUrl/docs"
        Write-Host "  Phoenix       : http://localhost:$phoenixPort"
        Write-Host "  Pruebas       : python scripts/run_scenarios.py"
        exit 0
    } catch {
        Write-Host -NoNewline "."
        Start-Sleep -Seconds 2
    }
}
Write-Host ""
Write-Host "La API no respondio a tiempo. Revisa: docker compose logs api"
exit 1
