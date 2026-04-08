<#
.SYNOPSIS
    Admin script for setting up, activating, testing, linting, and cleaning your project venv with uv.

.PARAMETER task
    What to do: 'active', 'make', 'test', 'lint', 'builddocs', 'servedocs', or 'clean'.
    Defaults to 'active'.
#>

param (
    [ValidateSet("active","make","test","lint","builddocs","servedocs","clean")]
    [string]$task = "active"
)

$ErrorActionPreference = 'Stop'

# Ensure we run from repo root
Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

# --- read exact Python version from .python-version ---
if (Test-Path ".\.python-version") {
    $pythonVersion = (Get-Content ".\.python-version" -ErrorAction Stop).Trim()
    if (-not $pythonVersion) {
        Throw ".python-version is empty. Write '3.14' in it."
    }
} else {
    Throw "Required file .python-version not found. Create it with '3.14' as it's only content."
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
        python -m pip install uv
        uv lock
        uv pip install -e ".[dev,docs]"
        pre-commit install
    }

    "test" {
        pytest
    }

    "lint" {
        ruff check . --fix --exit-non-zero-on-fix
        ruff format
        mypy .
    }

    "builddocs" {
        Write-Host "📚 Building documentation..." -ForegroundColor Cyan
        Push-Location docs
        try {
            sphinx-build -b html source build/html
            Write-Host "✅ Documentation built in docs/build/html/" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Documentation build failed: $_" -ForegroundColor Red
            exit 1
        }
        finally {
            Pop-Location
        }
    }

    "servedocs" {
        Write-Host "📚 Starting live documentation server..." -ForegroundColor Cyan
        Push-Location docs
        try {
            Write-Host "🌐 Documentation server will run at http://127.0.0.1:8000" -ForegroundColor Yellow
            Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
            sphinx-autobuild source build/html --host 127.0.0.1 --port 8000 --re-ignore ".*\..*"
        }
        catch {
            Write-Host "❌ Failed to start documentation server: $_" -ForegroundColor Red
            exit 1
        }
        finally {
            Pop-Location
        }
    }

    "clean" {
        Write-Host "🧹 Cleaning documentation artifacts..." -ForegroundColor Cyan
        if (Test-Path ".\docs\build") { Remove-Item ".\docs\build" -Recurse -Force }
        if (Test-Path ".\docs\source\api\generated") { Remove-Item ".\docs\source\api\generated" -Recurse -Force }

        . .\venv\Scripts\Activate.ps1
        pre-commit uninstall
        if ($env:VIRTUAL_ENV) {
            & "$env:VIRTUAL_ENV\Scripts\deactivate.bat" 2>$null
        }
        if (Test-Path .\venv) {
            Remove-Item .\venv -Recurse -Force
        }
        Write-Host "🧹 Clean complete." -ForegroundColor Green
    }

    default {
        Write-Error "Invalid task '$task'. Use active, make, test, lint, builddocs, servedocs, or clean."
        exit 1
    }
}

Pop-Location
