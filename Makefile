.ONESHELL:

.PHONY: all install test lint clean builddocs servedocs cleandocs deploy_docs

all: install

install:
	python -m venv ./venv
	venv/bin/python --version
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install poetry==2.2.1
	@. venv/bin/activate && \
	poetry install --with dev,docs && \
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

builddocs:
	@echo "📚 Building documentation..."
	cd docs && poetry run sphinx-build -b html source build/html
	@echo "✅ Documentation built in docs/build/html/"

servedocs:
	@echo "📚 Starting live documentation server..."
	cd docs && poetry run sphinx-autobuild source build/html --host 127.0.0.1 --port 8000 --re-ignore ".*\..*"
	@echo "🌐 Documentation server running at http://127.0.0.1:8000"

cleandocs:
	@echo "🧹 Cleaning documentation artifacts..."
	rm -rf docs/build/ docs/source/api/generated/
	@echo "✅ Documentation artifacts cleaned"

deploy_docs: builddocs
	@echo "🚀 Deploying documentation to gh-pages branch..."
	@git checkout gh-pages 2>/dev/null || git checkout -b gh-pages
	@rm -rf *
	@cp -r docs/build/html/* .
	@git add .
	@git commit -m "Update docs from $$(git log -1 --format='%h %s' --abbrev=10)" || echo "No changes to commit"
	@git push origin gh-pages
	@echo "✅ Documentation deployed to gh-pages"
	@echo "Switching back to previous branch..."
	@git checkout -
	@echo "Done! Your docs should be live at: https://captorab.github.io/openseries/"
