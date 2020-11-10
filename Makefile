
venv: requirements.txt
	python3 -m venv ./venv
	venv/bin/pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=${PWD} venv/bin/nosetests -vv --nologcapture --nocapture --with-timer ./

.PHONY: test

coverage:
	PYTHONPATH=${PWD} venv/bin/nosetests -vv --nologcapture --nocapture --with-timer --with-coverage ./
	PYTHONPATH=${PWD} venv/bin/coverage html -d coverage_html
	PYTHONPATH=${PWD} venv/bin/coverage-badge -o coverage.svg

clean:
	rm -rf venv
