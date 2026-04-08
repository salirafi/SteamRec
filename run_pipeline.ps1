[CmdletBinding()]
param(
    [string]$UploadDir = "",
    [string]$PythonCommand = "",
    [string]$BootstrapPython = "",
    [string]$MysqlCommand = "",
    [switch]$SkipInstall,
    [switch]$LaunchApp
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSCommandPath
$TempDir = Join-Path $RepoRoot "scripts\.tmp"
$DotEnvPath = Join-Path $RepoRoot ".env"
$VenvDir = Join-Path $RepoRoot ".venv"
$RequirementsPath = Join-Path $RepoRoot "requirements.txt"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }

        $parts = $trimmed.Split("=", 2)
        if ($parts.Length -ne 2) {
            continue
        }

        $name = $parts[0].Trim()
        $value = $parts[1].Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        Set-Item -Path "Env:$name" -Value $value
    }
}

function Require-Command([string]$CommandName) {
    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$CommandName' was not found in PATH."
    }
}

function Resolve-BootstrapPython([string]$PreferredCommand) {
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PreferredCommand)) {
        $candidates += $PreferredCommand
    }
    $candidates += @("python", "py")

    foreach ($candidate in $candidates | Select-Object -Unique) {
        if (Get-Command $candidate -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }

    throw "No Python interpreter was found for creating the virtual environment."
}

function Initialize-Venv(
    [string]$VenvPath,
    [string]$RequirementsFile,
    [string]$PreferredBootstrapPython,
    [switch]$SkipDependencyInstall
) {
    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    $bootstrap = Resolve-BootstrapPython $PreferredBootstrapPython

    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Step "Creating virtual environment at .venv"
        if ($bootstrap -eq "py") {
            & $bootstrap -3 -m venv $VenvPath
        } else {
            & $bootstrap -m venv $VenvPath
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment."
        }
    }

    if (-not $SkipDependencyInstall) {
        if (-not (Test-Path -LiteralPath $RequirementsFile)) {
            throw "requirements.txt was not found at $RequirementsFile"
        }

        Write-Step "Installing Python dependencies into .venv"
        & $venvPython -m pip install --upgrade pip | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to upgrade pip in the virtual environment."
        }

        & $venvPython -m pip install -r $RequirementsFile | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install requirements from $RequirementsFile"
        }
    }

    return $venvPython
}

function Normalize-SqlPath([string]$PathValue) {
    return ($PathValue -replace "\\", "/").TrimEnd("/")
}

function New-PatchedSqlFile(
    [string]$TemplatePath,
    [string]$OutputPath,
    [string]$ProdDbName,
    [string]$QueryDbName,
    [string]$UploadDirValue,
    [switch]$EnsureQueryDb
) {
    $content = Get-Content -LiteralPath $TemplatePath -Raw
    $backtick = [char]96

    $content = $content.Replace(
        "$backtick" + "steam-rec" + "$backtick",
        "$backtick" + $ProdDbName + "$backtick"
    )
    $content = $content.Replace(
        "$backtick" + "steam-rec-query" + "$backtick",
        "$backtick" + $QueryDbName + "$backtick"
    )
    $content = $content.Replace(
        "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads",
        (Normalize-SqlPath $UploadDirValue)
    )

    if ($EnsureQueryDb) {
        $createQueryDb = "CREATE DATABASE IF NOT EXISTS $backtick$QueryDbName$backtick;" + [Environment]::NewLine + [Environment]::NewLine
        $content = $createQueryDb + $content
    }

    Set-Content -LiteralPath $OutputPath -Value $content -Encoding UTF8
}

function Invoke-MysqlScript([string]$SqlPath, [string]$MysqlExe) {
    Get-Content -LiteralPath $SqlPath -Raw | & $MysqlExe `
        "--user=$env:STEAM_DB_USER" `
        "--password=$env:STEAM_DB_PASSWORD" `
        "--host=$env:STEAM_DB_HOST" `
        "--port=$env:STEAM_DB_PORT"

    if ($LASTEXITCODE -ne 0) {
        throw "mysql exited with code $LASTEXITCODE while running $SqlPath"
    }
}

Import-DotEnv $DotEnvPath

if ([string]::IsNullOrWhiteSpace($MysqlCommand)) {
    $MysqlCommand = if ($env:MYSQL_BIN) { $env:MYSQL_BIN } else { "mysql" }
}
if ([string]::IsNullOrWhiteSpace($UploadDir)) {
    $UploadDir = if ($env:MYSQL_UPLOAD_DIR) { $env:MYSQL_UPLOAD_DIR } else { "C:\ProgramData\MySQL\MySQL Server 8.0\Uploads" }
}
if ([string]::IsNullOrWhiteSpace($BootstrapPython)) {
    $BootstrapPython = if ($env:PYTHON_BOOTSTRAP_BIN) { $env:PYTHON_BOOTSTRAP_BIN } else { "" }
}

Require-Command $MysqlCommand

foreach ($requiredVar in @("STEAM_DB_USER", "STEAM_DB_PASSWORD", "STEAM_DB_HOST", "STEAM_DB_PORT")) {
    if ([string]::IsNullOrWhiteSpace((Get-Item -Path "Env:$requiredVar" -ErrorAction SilentlyContinue).Value)) {
        throw "Missing required environment variable: $requiredVar"
    }
}

$ProdDbName = if ($env:STEAM_DB_PROD_NAME) { $env:STEAM_DB_PROD_NAME } else { "steam-rec" }
$QueryDbName = if ($env:STEAM_DB_QUERY_NAME) { $env:STEAM_DB_QUERY_NAME } else { "steam-rec-query" }

$PythonCommand = Initialize-Venv -VenvPath $VenvDir -RequirementsFile $RequirementsPath -PreferredBootstrapPython $BootstrapPython -SkipDependencyInstall:$SkipInstall

New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
New-Item -ItemType Directory -Path $UploadDir -Force | Out-Null

$ProdSqlTemplate = Join-Path $RepoRoot "sql_script\load_production_tables.sql"
$RecSqlTemplate = Join-Path $RepoRoot "sql_script\load_rec_query_from_csv.sql"
$ProdSqlRuntime = Join-Path $TempDir "load_production_tables.runtime.sql"
$RecSqlRuntime = Join-Path $TempDir "load_rec_query_from_csv.runtime.sql"

New-PatchedSqlFile -TemplatePath $ProdSqlTemplate -OutputPath $ProdSqlRuntime -ProdDbName $ProdDbName -QueryDbName $QueryDbName -UploadDirValue $UploadDir
New-PatchedSqlFile -TemplatePath $RecSqlTemplate -OutputPath $RecSqlRuntime -ProdDbName $ProdDbName -QueryDbName $QueryDbName -UploadDirValue $UploadDir -EnsureQueryDb

Push-Location $RepoRoot
try {
    Write-Step "Preparing production CSV tables"
    & $PythonCommand ".\src\prepare_production_tables.py"
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_production_tables.py failed with exit code $LASTEXITCODE"
    }

    Write-Step "Copying production CSV tables to MySQL upload directory"
    Copy-Item -Path (Join-Path $RepoRoot "tables\production\*") -Destination $UploadDir -Recurse -Force

    Write-Step "Loading production tables into MySQL"
    Invoke-MysqlScript -SqlPath $ProdSqlRuntime -MysqlExe $MysqlCommand

    Write-Step "Preparing recommender matrices"
    & $PythonCommand ".\src\prepare_recommender_matrices.py"
    if ($LASTEXITCODE -ne 0) {
        throw "prepare_recommender_matrices.py failed with exit code $LASTEXITCODE"
    }

    Write-Step "Copying recommender CSV tables to MySQL upload directory"
    Copy-Item -Path (Join-Path $RepoRoot "tables\rec_matrices\*") -Destination $UploadDir -Recurse -Force

    Write-Step "Loading recommender query tables into MySQL"
    Invoke-MysqlScript -SqlPath $RecSqlRuntime -MysqlExe $MysqlCommand

    if ($LaunchApp) {
        Write-Step "Launching Flask app"
        & $PythonCommand ".\app.py"
        if ($LASTEXITCODE -ne 0) {
            throw "app.py failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host ""
        Write-Host "Pipeline completed successfully." -ForegroundColor Green
        Write-Host "Run '.\run_app.ps1' to start the Flask app." -ForegroundColor Green
    }
}
finally {
    Pop-Location
}
