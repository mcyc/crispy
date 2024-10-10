from setuptools import setup, find_packages

setup(
    name="astro-crispy",
    version='2.0.0',
    description="Computational Ridge Identification with SCMS for Python",
    long_description=long_description,
    author="Michael Chun-Yuan Chen",
    author_email='mkid.chen@gmail.com',
    url='https://github.com/mcyc/crispy',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'astropy>=4.0',
        'scikit-image>=0.16.0',
        'scikit-learn>=0.22.0',
        'cuml>=0.15.0',
    ],
    extras_require={
        'cpu': ['scikit-learn>=0.22.0'],  # Allows CPU fallback if GPU is not available
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    license='LGPLv3',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)