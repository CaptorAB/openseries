venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==1.8.4

install:
	rm -f poetry.lock
	poetry install --with dev
	pre-commit install

test:
	poetry run coverage run -m pytest --verbose --capture=no
	poetry run coverage xml --quiet
	poetry run coverage report
	poetry run genbadge coverage --silent --local --input-file coverage.xml --output-file coverage.svg

lint:
	poetry run ruff check . --fix --exit-non-zero-on-fix
	poetry run ruff format
	poetry run mypy .

clean:
	rm -rf venv
	rm -f poetry.lock
