venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/pip install poetry==1.4.0
	poetry install --with dev

test:
	PYTHONPATH=${PWD} poetry run coverage run -m pytest --verbose --capture=no --durations=20 --durations-min=2.0 ./
	PYTHONPATH=${PWD} poetry run coverage report -m

.PHONY: test
