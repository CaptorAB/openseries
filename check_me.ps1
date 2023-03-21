param (
    [string]$task = "test"
)

$env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
.\venv\Scripts\activate
poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0 --store-durations
poetry run coverage report -m
poetry run coverage-badge -o coverage.svg -f
if ($task -eq "test") {
    # run commands for test task
    poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0 --store-durations
    poetry run coverage report -m
    poetry run coverage-badge -o coverage.svg -f
} elseif ($task -eq "lint") {
    # run commands for lint task
    poetry run coverage run flake8 .
} else {
    # invalid task argument
    Write-Host "Only test or lint are allowed as tasks"
}
