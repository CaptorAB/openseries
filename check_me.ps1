$env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
.\venv\Scripts\activate
poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0 --store-durations
poetry run coverage report -m
poetry run coverage-badge -o coverage.svg -f
