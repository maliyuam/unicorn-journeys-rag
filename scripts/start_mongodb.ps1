# Starts the user-level MongoDB instance for Unicorn Journeys RAG.
# No admin rights, no Windows service — mongod runs as your user from
# %LOCALAPPDATA%\UnicornRAG — deliberately outside the repo, so a synced
# folder (OneDrive, Dropbox) never tries to replicate the database files.
# Safe to run repeatedly: does nothing if mongod is already listening.

$base = Join-Path $env:LOCALAPPDATA "UnicornRAG"
$mongod = Get-ChildItem -Path $base -Filter "mongod.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $mongod) {
    Write-Error "mongod.exe not found under $base — see README (MongoDB setup)."
    exit 1
}

$dataDir = Join-Path $base "data"
$logDir = Join-Path $base "logs"
New-Item -ItemType Directory -Force $dataDir, $logDir | Out-Null

$listening = Test-NetConnection -ComputerName 127.0.0.1 -Port 27017 -InformationLevel Quiet -WarningAction SilentlyContinue
if ($listening) {
    Write-Host "MongoDB already running on 127.0.0.1:27017"
    exit 0
}

Start-Process -FilePath $mongod.FullName -WindowStyle Hidden -ArgumentList @(
    "--dbpath", $dataDir,
    "--logpath", (Join-Path $logDir "mongod.log"),
    "--port", "27017",
    "--bind_ip", "127.0.0.1"
)
Start-Sleep -Seconds 3
$listening = Test-NetConnection -ComputerName 127.0.0.1 -Port 27017 -InformationLevel Quiet -WarningAction SilentlyContinue
if ($listening) {
    Write-Host "MongoDB started on 127.0.0.1:27017 (data: $dataDir)"
} else {
    Write-Error "mongod did not come up — check $logDir\mongod.log"
    exit 1
}
