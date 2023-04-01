#!/bin/bash

output=$(poetry run flake8 . 2>&1)
if [ $? -eq 0 ]; then
  echo -e "\e[32mFlake8 linting is OK\e[0m"
else
  echo "$output"
fi
poetry run mypy .
