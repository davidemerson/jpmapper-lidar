# build your environment

- Clone this repo.
- Get Miniconda, https://docs.conda.io/en/latest/miniconda.html
- Launch Anaconda Prompt
- Create your environment in Miniconda, `conda create -n jpmapper python=3.11`
- Activate the project, `conda activate jpmapper`
- Get some core dependencies, `conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer`
- Get some extras from pip, `pip install pytest pytest-cov ruff mypy pre-commit`
- Establish an install of jpmapper, `pip install -e .`
- See if stuff works `jpmapper --help`
- Run a quick test, `pytest -q`