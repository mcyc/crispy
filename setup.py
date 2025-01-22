import tomli
from setuptools import setup, find_packages

# Read metadata from pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject_data = tomli.load(f)

project_metadata = pyproject_data["project"]

setup(
    name=project_metadata["name"],
    version=project_metadata["version"],
    author=project_metadata["authors"][0]["name"],
    author_email=project_metadata["authors"][0]["email"],
    description=project_metadata["description"],
    long_description=open(project_metadata["readme"], "r").read(),
    long_description_content_type="text/markdown",
    url=project_metadata["urls"]["Source"],
    packages=find_packages(),
    install_requires=project_metadata["dependencies"],
    classifiers=project_metadata["classifiers"],
    python_requires=project_metadata["requires-python"],
)
