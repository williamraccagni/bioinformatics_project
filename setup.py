import os
import re
# To use a consistent encoding
from codecs import open as copen

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with copen(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with copen(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("bioinformatica", "__version__.py")

test_deps =[
    "pytest",
    "pytest-cov",
    "coveralls",
    "validate_version_code",
    "codacy-coverage",
    "scipy",
    "pandas",
    "numpy",
    "epigenomic-dataset",
    "matplotlib",
    "setuptools",
    "scikit-learn",
    "Boruta",
    "keras-bed-sequence",
    "tensorflow",
    "keras-mixed-sequence",
    "ucsc-genomes-downloader",
    "keras-tqdm",
    "minepy",
    "imblearn",
    "matplotlib",
    "seaborn",
    "barplots"
]

extras = {
    'test': test_deps,
}

setup(
    name='bioinformatica',
    version=__version__,
    description="repository progetto bioinformatica",
    long_description=long_description,
    url="https://github.com/steckk/bioinformatica",
    author="federico.stacchietti@gmail.com",
    author_email="steckk",
    # Choose your license
    license='MIT',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    tests_require=test_deps,
    # Add here the package dependencies
    install_requires=[],
    extras_require=extras,
)