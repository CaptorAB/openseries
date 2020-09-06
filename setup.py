from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="OpenSeries",
    version="0.1.1",
    packages=["OpenSeries"],
    data_files=[("", ["openseries.json", "plotly_captor_logo.json", "plotly_layouts.json"]),
                ("tests", ["key_value_table_with_relative.json", "series.json"])],
    url="https://github.com/CaptorAB/OpenSeries",
    license="BSD License",
    author="karrmagadgeteer2",
    author_email="martin.karrin@captor.se",
    description="A module for analyzing financial time series",
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
