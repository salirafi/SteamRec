[CmdletBinding()]
param(
    [string]$PythonCommand = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSCommandPath
$DotEnvPath = Join-Path $RepoRoot ".env"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

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

Import-DotEnv $DotEnvPath

if ([string]::IsNullOrWhiteSpace($PythonCommand)) {
    $PythonCommand = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } elseif ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
}

if (-not (Get-Command $PythonCommand -ErrorAction SilentlyContinue)) {
    throw "Required command '$PythonCommand' was not found in PATH."
}

Push-Location $RepoRoot
try {
    & $PythonCommand ".\app.py"
    if ($LASTEXITCODE -ne 0) {
        throw "app.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
