param (
    [string]$task = "active"
)

if ($task -eq "active")
{
    if ($null -ne $env:PYTHONPATH)
    {
        if (-not $env:PYTHONPATH -match [regex]::Escape($PWD))
        {
            $env:PYTHONPATH += ";$PWD"
            Write-Output "PYTHONPATH changed. It is '$( $env:PYTHONPATH )'"
        }
        else
        {
            Write-Output "PYTHONPATH not changed. It is '$( $env:PYTHONPATH )'"
        }
    }
    else
    {
        $env:PYTHONPATH = $PWD
        Write-Output "PYTHONPATH set to: $( $env:PYTHONPATH )"
    }
    .\venv\Scripts\activate
}
elseif ($task -eq "make")
{
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    python -m venv ./venv
    if ($null -ne $env:PYTHONPATH)
    {
        if (-not $env:PYTHONPATH -match [regex]::Escape($PWD))
        {
            $env:PYTHONPATH += ";$PWD"
            Write-Output "PYTHONPATH changed. It is '$( $env:PYTHONPATH )'"
        }
        else
        {
            Write-Output "PYTHONPATH not changed. It is '$( $env:PYTHONPATH )'"
        }
    }
    else
    {
        $env:PYTHONPATH = $PWD
        Write-Output "PYTHONPATH set to: $( $env:PYTHONPATH )"
    }
    .\venv\Scripts\activate
    python.exe -m pip install --upgrade pip
    pip install poetry==1.8.2
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
    poetry install --with dev
    poetry run pre-commit install
}
elseif ($task -eq "test")
{
    poetry run coverage run -m pytest --verbose --capture=no
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
}
elseif ($task -eq "lint")
{
    poetry run ruff check . --fix --exit-non-zero-on-fix
    poetry run mypy .
}
elseif ($task -eq "clean")
{
    pre-commit uninstall
    deactivate
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
}
else
{
    Write-Host "Only active, make, test, clean or lint are allowed as tasks"
}
