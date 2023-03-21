venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install poetry==1.4.0
	poetry install --with dev
	pre-commit install

test:
	PYTHONPATH=${PWD} poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0 --store-durations
	PYTHONPATH=${PWD} poetry run coverage report -m
	PYTHONPATH=${PWD} poetry run coverage-badge -o coverage.svg -f

lint:
	PYTHONPATH=${PWD} poetry run flake8 .

.PHONY: test
