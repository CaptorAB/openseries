.ONESHELL:

.PHONY: all install test lint clean builddocs servedocs cleandocs

all: install

install:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install uv
	@. venv/bin/activate && \
	uv lock && \
	uv pip install -e ".[dev,docs]" && \
	pre-commit install

test:
	pytest

lint:
	ruff check . --fix --exit-non-zero-on-fix
	ruff format
	mypy .

clean:
	@. venv/bin/activate && \
	pre-commit uninstall && \
	rm -rf venv

builddocs:
	@echo "📚 Building documentation..."
	cd docs && sphinx-build -b html source build/html
	@echo "✅ Documentation built in docs/build/html/"

servedocs:
	@echo "📚 Starting live documentation server..."
	cd docs && sphinx-autobuild source build/html --host 127.0.0.1 --port 8000 --re-ignore ".*\..*"
	@echo "🌐 Documentation server running at http://127.0.0.1:8000"

cleandocs:
	@echo "🧹 Cleaning documentation artifacts..."
	rm -rf docs/build/ docs/source/api/generated/
	@echo "✅ Documentation artifacts cleaned"
