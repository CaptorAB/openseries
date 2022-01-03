.PHONY: test

venv: requirements.txt
	python3 -m venv ./venv
	venv/bin/pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/nosetests -v --nologcapture --nocapture --with-timer ./

lint:
	python -m flake8 openseries setup.py

fix:
	python -m black openseries setup.py

dist:
	python setup.py sdist

upload: clean dist
	twine upload dist/*

coverage:
	PYTHONPATH=${PWD} venv/bin/nosetests -vv --nologcapture --nocapture --with-timer --with-coverage ./
	PYTHONPATH=${PWD} venv/bin/coverage html -d coverage_html
	PYTHONPATH=${PWD} venv/bin/coverage-badge -o coverage.svg

clean:
	rm -rf dist
	rm -rf openseries.egg-info
