.ONESHELL:

.PHONY: all install test lint clean

all: install

install:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==2.2.1
	@. venv/bin/activate && \
	poetry install --with dev && \
	poetry run pre-commit install

test:
	poetry run pytest

lint:
	poetry run ruff check . --fix --exit-non-zero-on-fix
	poetry run ruff format
	poetry run mypy .

clean:
	@. venv/bin/activate && \
	pre-commit uninstall && \
	rm -rf venv && \
	rm -f poetry.lock
