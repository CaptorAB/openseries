.PHONY: test

venv: requirements.txt
	python3 -m venv ./venv
	export PYTHONPATH=$PYTHONPATH:${PWD}
	source venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose ./
	PYTHONPATH=${PWD} venv/bin/coverage report

lint:
	PYTHONPATH=${PWD} venv/bin/flake8 ./

fix:
	PYTHONPATH=${PWD} venv/bin/black ./

dist:
	PYTHONPATH=${PWD} venv/bin/python setup.py sdist

upload: clean dist
	PYTHONPATH=${PWD} venv/bin/twine upload dist/*

clean:
	rm -rf dist
	rm -rf openseries.egg-info
	rm -rf venv
