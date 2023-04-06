param (
    [string]$task = "active"
)

if ($task -eq "active") {
    # run commands to activate virtual environment
    $env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
    .\venv\Scripts\activate
}
elseif ($task -eq "make") {
    # make virtual environment
    python -m venv ./venv
    $env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
    .\venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install poetry==1.4.2
    poetry install --with dev
    pre-commit install
}
elseif ($task -eq "test") {
    # run tests and report coverage
    poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
}
elseif ($task -eq "lint") {
    # run lint and typing checks
    $lintresult = poetry run flake8 .
    if ($LASTEXITCODE -eq 0) {
        Write-Host -ForegroundColor Green "Flake8 linting is OK"
    }
    else {
        Write-Host $lintresult
    }
    poetry run mypy .
    poetry run pylint ./openseries/* ./tests/*
}
elseif ($task -eq "clean") {
    # remove virtual environment to start over
    Remove-Item -Path ".\venv" -Recurse -Force
}
else {
    # invalid task argument
    Write-Host "Only active, make, test, clean or lint are allowed as tasks"
}
