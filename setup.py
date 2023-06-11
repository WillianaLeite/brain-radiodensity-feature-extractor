# Always prefer setuptools over distutils
from setuptools import setup

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read all requeriments
with open('requirements.txt') as f:
    required = f.read().splitlines()

# This call to setup() does all the work
setup(
    name="brain-radiodensity-feature-extractor",
    version="0.1.1",
    description="A simple library that extracts brain features from a Brain Tomography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://brain-radiodensity-feature-extractor.readthedocs.io/",
    author="Williana Leite",
    author_email="willianaluziasousaleite@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["brain_feature_extractor"],
    include_package_data=True,
    install_requires=required
)