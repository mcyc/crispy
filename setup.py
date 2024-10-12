from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="astro-crispy", # Replace with your own username
    version="1.3.0",
    author="Michael Chun-Yuan Chen",
    author_email="chen.m@queensu.ca",
    description="Computational Ridge Identification with SCMS for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcyc/crispy",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'scikit-learn',
        'scikit-image',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
