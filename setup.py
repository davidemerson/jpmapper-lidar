from setuptools import setup, find_packages

setup(
    name="jpmapper",
    packages=find_packages(include=["jpmapper", "jpmapper.*"]),
    # Other metadata is read from pyproject.toml
)
