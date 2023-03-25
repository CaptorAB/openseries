venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install poetry==1.4.0
    export PYTHONPATH=$PYTHONPATH:${PWD}
    source venv/bin/activate
	poetry install --with dev
	pre-commit install

active:
    export PYTHONPATH=$PYTHONPATH:${PWD}
    source venv/bin/activate

test:
	PYTHONPATH=${PWD} poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0
	PYTHONPATH=${PWD} poetry run coverage report -m
	PYTHONPATH=${PWD} poetry run coverage-badge -o coverage.svg -f

lint:
	PYTHONPATH=${PWD} poetry run flake8 .
	PYTHONPATH=${PWD} poetry run mypy .

.PHONY: test
