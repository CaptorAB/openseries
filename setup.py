from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="openseries",
    version="0.1.8",
    packages=find_packages(),
    package_data={"OpenSeries": ["*.json"], ".": ["*.json"]},
    install_requires=["jsonschema", "numpy", "pandas<=1.0.4", "plotly", "python-dateutil",
                      "requests", "scipy", "statsmodels"],
    url="https://github.com/CaptorAB/OpenSeries",
    license="BSD License",
    author="karrmagadgeteer2",
    author_email="martin.karrin@captor.se",
    description="A package for analyzing financial time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
