$env:PYTHONPATH = "$env:PYTHONPATH;$pwd"
.\venv\Scripts\activate
.\venv\Scripts\coverage run -m pytest --verbose --durations=20 --durations-min=2.0 ./
.\venv\Scripts\coverage report -m