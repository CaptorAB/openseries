#!/bin/bash
$PYTHON -m pip install --no-deps --ignore-installed .
$PYTHON -m coverage run -m pytest --verbose --capture=no
$PYTHON -m coverage report -m
