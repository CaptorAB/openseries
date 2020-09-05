from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='OpenSeries',
    version='0.1.0',
    packages=['OpenSeries'],
    url='https://github.com/CaptorAB/OpenSeries',
    license='BSD-3-Clause License',
    author='karrmagadgeteer2',
    author_email='martin.karrin@captor.se',
    description='A module for analyzing financial time series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
