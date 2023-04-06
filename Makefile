venv:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==1.4.2

install:
	poetry install --with dev
	pre-commit install

test:
	PYTHONPATH=${PWD} poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0
	PYTHONPATH=${PWD} poetry run coverage report -m
	PYTHONPATH=${PWD} poetry run coverage-badge -o coverage.svg -f

lint:
	PYTHONPATH=${PWD} poetry run flake8 .
	PYTHONPATH=${PWD} poetry run mypy .
	PYTHONPATH=${PWD} poetry run pylint ./openseries/* ./tests/*

clean:
	deactivate
	rm -rf venv
