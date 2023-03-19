$env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
.\venv\Scripts\activate
poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0
poetry run coverage report -m
