param (
    [string]$task = "active"
)

if ($task -eq "active") {
    $env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
    .\venv\Scripts\activate
}
elseif ($task -eq "make") {
    # remove old environment if present and create new environment
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    python -m venv ./venv
    $env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
    .\venv\Scripts\activate
    python.exe -m pip install --upgrade pip
    pip install poetry==1.8.2
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
    poetry install --with dev
    poetry run pre-commit install
}
elseif ($task -eq "test") {
    # run tests and report coverage
    poetry run coverage run -m pytest --verbose --capture=no
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
}
elseif ($task -eq "lint") {
    # run lint and typing checks
    poetry run ruff check . --fix --exit-non-zero-on-fix
    poetry run mypy .
}
elseif ($task -eq "clean") {
    # remove virtual environment and lock file to start over
    pre-commit uninstall
    deactivate
    Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path 'poetry.lock' -Force -ErrorAction SilentlyContinue
}
else {
    # invalid task argument
    Write-Host "Only active, make, test, clean or lint are allowed as tasks"
}
