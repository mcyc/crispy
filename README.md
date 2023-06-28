## CRISPy
Computational Ridge Identification with SCMS for Python

This code is a python implementation of the generalized Subspace Constrained Mean Shift (SCMS) algorithm, translated and modified from the [R-code](https://sites.google.com/site/yenchicr/algorithm) developed by [Yen-Chi Chen](http://faculty.washington.edu/yenchic/) (University of Washington).

Please cite the following papers when using the code:
1. Ozertem, Umut, and Deniz Erdogmus. "Locally Defined Principal Curves and Surfaces." The Journal of Machine Learning Research 12 (2011): 1249-1286.
2. Chen, Yen-Chi, Christopher Genovese, Christopher R. Genovese, and Larry Wasserman. "Generalized Mode and Ridge Estimation." (2014): arXiv :1406.1803
3. Chen, Yen-Chi, Shirley Ho, Peter E. Freeman, Christopher R. Genovese, and Larry Wasserman. "Cosmic Web Reconstruction through Density Ridges: Method and Algorithm." MNRAS 454 1140 (2015) arXiv: 1501.05303
4. Chen, Michael Chun-Yuan, et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" [ApJ 891 (2020) 84](https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract)

## Installation

To install the latest version of ```CRISPy``` from this repository, run:

```
python setup.py install
```

To pip install a 'stable' release, run:
```
pip install astro-crispy
```


## Minimum Working Example

To find density ridges from a .fits image (e.g., a cube) with the name ```fname``` and write the results to a .txt file with the name ```savename```, run:

```
From crispy import image_ridge_find as irf 

G = irf.run(fname, h=h, thres=thres)
irf.write_output(G, savename)
```

Where ```h``` and ```thres``` are the smoothing length and the intensity thresholds, respectively. For a start, an ```h``` value that's comparable to the resolution element of your data (e.g., the FWHM beamsize) is recommended. The ```thres``` value should be set well above the noise level of your data, where the signal of your structures is reasonably robust.

