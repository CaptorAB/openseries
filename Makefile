venv:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==1.7.1

install:
	rm -f poetry.lock
	poetry install --with dev
	pre-commit install

test:
	poetry run coverage run -m pytest --verbose --capture=no
	poetry run coverage report -m
	poetry run coverage-badge -o coverage.svg -f

lint:
	poetry run ruff check . --fix --exit-non-zero-on-fix
	poetry run black ./openseries/*.py ./tests/*.py
	poetry run mypy .

clean:
	rm -rf venv
	rm -f poetry.lock
