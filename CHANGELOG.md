# Changelog

For a long time I have not kept a log of the changes implemented in the different versions of the openseries package. In this file I am attempting to rectify this somewhat. However, unfortunately I do not have the resources to issue any form of guarantee that this log will cover all changes, and I will not attempt to go back very far in history.

## Version [0.5.7] - 2022-07-24

Fixed rolling correlation, added beta attribute and rolling beta for OpenFrame and associated tests.

## Version [0.5.5] - 2022-07-17

Removed log returns everywhere. Removed keyvaluetable and reduced use of date_fix function. Improved test coverage further and will leave at this level for now.

## Version [0.5.2] - 2022-07-15

Fixed so that ratios based on geometric returns will use arithmetic return instead to avoid some failures. The geo_ret functions will now raise and exception on initial zeroes and on negative values. Improved test coverage and also added missing PEP604 type hints.

## Version [0.5.0] - 2022-07-12

This version can only run on Python version 3.10 due to the implementation of type hints following [PEP 604](https://peps.python.org/pep-0604/).

## Version [0.4.1] - 2022-07-12

This version is backwards compatible only from Python version 3.8 due to the implementation of [PEP 589](https://peps.python.org/pep-0589/).

## Version [0.4.0] - 2022-06-30

This version is backwards compatible to Python version 3.6 and works up to version 3.10, docstrings have been improved and deprecation warnings fixed.

## Version [0.3.8] - 2022-05-30

This is the first draft version to work with Python version 3.10. It runs but with several deprecation warnings primarily from Pandas. Prior to this version openseries was not compatible with Python version 3.10.
