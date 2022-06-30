
venv:
	python3 -m venv ./venv
	. venv/bin/activate
	PYTHONPATH=${PWD} venv/bin/python --version
	PYTHONPATH=${PWD} venv/bin/pip install --upgrade pip
	PYTHONPATH=${PWD} venv/bin/pip install --upgrade -r requirements.txt
	PYTHONPATH=${PWD} venv/bin/pip --version

test:
	PYTHONPATH=${PWD} venv/bin/coverage run -m pytest --verbose ./
	PYTHONPATH=${PWD} venv/bin/coverage report

lint:
	PYTHONPATH=${PWD} venv/bin/flake8 ./

fix:
	PYTHONPATH=${PWD} venv/bin/black ./

dist:
	PYTHONPATH=${PWD} venv/bin/python setup.py sdist

upload:
	PYTHONPATH=${PWD} venv/bin/twine upload dist/*

clean:
	rm -rf dist
	rm -rf openseries.egg-info
	rm -rf venv

.PHONY: test
