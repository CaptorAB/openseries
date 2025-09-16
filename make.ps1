<#
.SYNOPSIS
    Admin script for setting up, activating, testing, linting, and cleaning your project venv with Poetry.

.PARAMETER task
    What to do: 'active', 'make', 'test', 'lint', or 'clean'.
    Defaults to 'active'.
#>

param (
    [ValidateSet("active","make","test","lint","clean")]
    [string]$task = "active"
)

$ErrorActionPreference = 'Stop'

# Pin your Poetry version here:
[string]$poetryVersion = '2.2.0'

# Ensure we run from repo root
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# --- read exact Python version from .python-version ---
if (Test-Path ".\.python-version") {
    $pythonVersion = (Get-Content ".\.python-version" -ErrorAction Stop).Trim()
    if (-not $pythonVersion) {
        Throw ".python-version is empty. Write '3.13' in it."
    }
} else {
    Throw "Required file .python-version not found. Create it with '3.13' as it's only content."
}

function Ensure-PythonPath {
    if ($env:PYTHONPATH) {
        if (-not ($env:PYTHONPATH -match [regex]::Escape($PWD))) {
            $env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
            Write-Output "`nPYTHONPATH updated to: $env:PYTHONPATH"
        } else {
            Write-Output "`nPYTHONPATH already includes project root."
        }
    } else {
        $env:PYTHONPATH = $PWD
        Write-Output "`nPYTHONPATH set to: $env:PYTHONPATH"
    }
}

switch ($task) {
    "active" {
        . .\venv\Scripts\Activate.ps1
        Ensure-PythonPath
        Write-Output "`nUsing Python in venv '$(Split-Path $env:VIRTUAL_ENV -Leaf)':"
        python --version
    }

    "make" {
        # remove any existing venv
        if (Test-Path .\venv) { Remove-Item .\venv -Recurse -Force }

        # if pyenv exists, pin it; otherwise verify system Python
        if (Test-Path "$env:USERPROFILE\.pyenv") {
            pyenv global  $pythonVersion
            pyenv local   $pythonVersion
            Write-Output "Set pyenv global & local to Python $pythonVersion."
        } else {
            $sysVer = (& python --version 2>&1) -replace 'Python ', ''
            if ($sysVer -eq $pythonVersion) {
                Write-Output "System Python $sysVer matches required $pythonVersion."
            } else {
                Write-Warning "System Python is $sysVer; expected $pythonVersion. Please install or use pyenv-win."
            }
        }

        # create & activate venv
        python -m venv .\venv
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Virtual environment created." -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to create virtual environment." -ForegroundColor Red
            exit 1
        }
        . .\venv\Scripts\Activate.ps1

        Ensure-PythonPath
        Write-Output "`nUsing Python in venv '$(Split-Path $env:VIRTUAL_ENV -Leaf)':"
        python --version

        # install tooling & deps
        python -m pip install --upgrade pip
        python -m pip install "poetry==$poetryVersion"
        if (Test-Path 'poetry.lock') {
            Remove-Item 'poetry.lock' -Force -ErrorAction SilentlyContinue
        }
        poetry install --with dev
        poetry run pre-commit install
    }

    "test" {
        poetry run pytest
    }

    "lint" {
        poetry run ruff check . --fix --exit-non-zero-on-fix
        poetry run ruff format
        poetry run mypy .
    }

    "clean" {
        . .\venv\Scripts\Activate.ps1
        poetry run pre-commit uninstall
        if ($env:VIRTUAL_ENV) {
            & "$env:VIRTUAL_ENV\Scripts\deactivate.bat" 2>$null
        }
        if (Test-Path .\venv) {
            Remove-Item .\venv -Recurse -Force
        }
        if (Test-Path 'poetry.lock') {
            Remove-Item 'poetry.lock' -Force -ErrorAction SilentlyContinue
        }
        Write-Output "🧹 Clean complete."
    }

    default {
        Write-Error "Invalid task '$task'. Use active, make, test, lint, or clean."
        exit 1
    }
}

Pop-Location
