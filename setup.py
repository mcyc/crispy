import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astro-crispy", # Replace with your own username
    version="0.1.0",
    author="Michael Chun-Yuan Chen",
    author_email="mcychen@uvic.ca",
    description="Computational Ridge Identification with SCMS for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcyc/crispy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)