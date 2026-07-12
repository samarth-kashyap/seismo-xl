"""Legacy setup for editable installs. Primary config in pyproject.toml."""
# File: setup.py
from setuptools import find_packages, setup

setup(
    name="seismo-xl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
) ;
