.ONESHELL:

UV_VERSION ?= 0.11.21
PIP_AUDIT_VERSION ?= 2.10.0

.PHONY: all install update test lint audit clean builddocs servedocs cleandocs

all: install

install:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install uv==$(UV_VERSION)
	@. venv/bin/activate && \
	uv lock && \
	uv sync --active --locked --extra dev --extra docs && \
	pre-commit install

update:
	@. venv/bin/activate && \
	python -m pip install --upgrade pip && \
	pip install --upgrade uv==$(UV_VERSION) && \
	uv lock --upgrade && \
	uv sync --active --locked --extra dev --extra docs

test:
	pytest

lint:
	ruff check . --fix --exit-non-zero-on-fix
	ruff format
	mypy .

audit:
	@. venv/bin/activate && \
	uv lock --check && \
	uv export --locked --extra dev --no-emit-project -o requirements-audit.txt && \
	uvx pip-audit==$(PIP_AUDIT_VERSION) -r requirements-audit.txt && \
	rm -f requirements-audit.txt

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
