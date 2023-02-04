import setuptools
from Cython.Build import cythonize
# import library to be included for cython
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astro-crispy", # Replace with your own username
    version="2.0.0",
    author="Michael Chun-Yuan Chen",
    author_email="chen.m@queensu.ca",
    description="Computational Ridge Identification with SCMS for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcyc/crispy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.3',
    ext_modules = cythonize("crispy/gaussian.pyx"),
    include_dirs=[numpy.get_include()],
    requires = ["setuptools", "wheel", "Cython"]
)