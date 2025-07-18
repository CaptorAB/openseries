.ONESHELL:

.PHONY: all install test lint clean

all: install

install:
	python3 -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==2.1.3
	@. venv/bin/activate && \
	poetry install --with dev && \
	poetry run pre-commit install

test:
	poetry run pytest -n auto --dist loadscope --cov=openseries --cov-report=term --cov-report=term-missing

lint:
	poetry run ruff check . --fix --exit-non-zero-on-fix
	poetry run ruff format
	poetry run mypy .

clean:
	@. venv/bin/activate && \
	pre-commit uninstall && \
	rm -rf venv && \
	rm -f poetry.lock
