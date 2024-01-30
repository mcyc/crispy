## CRISPy
Computational Ridge Identification with SCMS for Python

This code is a python implementation of the generalized Subspace Constrained Mean Shift (SCMS) algorithm, translated and modified from the [R-code](https://sites.google.com/site/yenchicr/algorithm) developed by [Yen-Chi Chen](http://faculty.washington.edu/yenchic/) (University of Washington) to run on gridded images.


Author: [Mike Chen](mailto:chen.m@queensu.ca)

Please cite the following papers when using the code for a publication:
1. Chen, Michael Chun-Yuan, et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?" [ApJ 891 (2020) 84](https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract)
2. Chen, Yen-Chi, Shirley Ho, Peter E. Freeman, Christopher R. Genovese, and Larry Wasserman. "Cosmic Web Reconstruction through Density Ridges: Method and Algorithm." MNRAS 454 1140 (2015) arXiv: 1501.05303
3. Chen, Yen-Chi, Christopher Genovese, Christopher R. Genovese, and Larry Wasserman. "Generalized Mode and Ridge Estimation." (2014): arXiv :1406.1803
4. Ozertem, Umut, and Deniz Erdogmus. "Locally Defined Principal Curves and Surfaces." The Journal of Machine Learning Research 12 (2011): 1249-1286.


## Installation

To install the latest version of ```CRISPy``` from this repository, run:

```
python setup.py install
```

To pip install a 'stable' release, run:
```
pip install astro-crispy
```

Please note that the last 'stable' version is rather out of date, and installing the latest version of this repository is recommended.


## Minimum Working Example

To find density ridges from a .fits image (e.g., a cube) with the name ```fname``` and write the results to a .txt file with the name ```savename```, run:

```
from crispy import image_ridge_find as irf 

G = irf.run(fname, h=h, thres=thres)
irf.write_output(G, savename)
```

Where ```h``` and ```thres``` are the smoothing length and the intensity thresholds, respectively. For a start, an ```h``` value comparable to your data's resolution element (e.g., the FWHM beamsize in pixel units) is recommended. The ```thres``` value should be set well above the noise level of your data, where the signal of your structures is reasonably robust.

To grid ```CRISPy``` results onto the native grid of the input image, run the following:

```
from crispy import grid_ridge as gr

gr.grid_skel(readFile, imgFile, writeFile)
```

where ```readFile``` is the .txt results of the ```CRISPy``` run, ```imgFile``` is the .fits image file from which ```CRISPy``` was ran on, and ```writeFile``` is the name of the gridded result in .fits format.

To prune the gridded skeletons, run the following:

```
from crispy.pruning import Skeleton

skel_obj = Skeleton.Skeleton(skelFile)
skel_obj.prune_from_scratch()
skel_obj.save_pruned_skel(spineFile, overwrite=True)
```

where ```skelFile``` is the .fits image file of the gridded result and ```spineFile``` is the name of the pruned spine image to be saved.


## Tips on Input Parameters 

#### Smoothing Bandwidth (h)

The smoothing bandwidth, ```h```, is the effective image resolution in pixel (e.g., voxel) units. For Nyquist-sampled images, I'd recommend starting with ```h=2```. Larger values are good for suppressing the image noise. However, if the desired smoothing bandwidth is greater than 6, it would be computationally more efficient to convolve the image first, downsample it (Nyquist-sampled still), and then run the new image with an h value around 2. For astronomical use, please remember that the smoothing bandwidth applies in all dimensions, including the spectral axis. If desired, the ```crdScaling``` parameter can be used to recale the effective h relative to individual axis.

#### Density Threshold (thres)

The density threshold, ```thres```, is the minimal value for a pixel to be considered data. Pixels with lower values will be masked out to save computational time and reduce noisy artefacts.

#### Threshold Mask (overmask)

The ```overmask``` parameter allows the user to provide a custom mask to specify which pixels to be included in a run as data, rather than using the one defined by ```thres```.

#### Convergence Criterion (eps)

The convergence criterion ```eps``` is the precision of the ridge desired. Higher precisions (i.e., lower eps values) will take longer for the walkers to converge. In general, ```eps = 0.01``` is recommended. For test runs or some science cases, however, eps = 0.1 will likely suffice.

#### Maximum Number of Iterations (maxT)

Some walkers may take exceptionally long to converge, depending on the image and mask qualities. The ```maxT``` parameter sets the maximum number of iterations before the run terminates. Depending on the image, the maxium number of iterations may need to be much higher than the default (1000) for a reliable result. The ```converge_frac``` can be used in complement with ```maxT``` to ensure a run terminates on a timely basis.

#### Walker Convergence Fraction (converge_frac)

The ```converge_frac``` parameter sets the threshold to terminate the run based on the fraction of walkers converged (1 being 100%). Since some walkers will inevitably converge exceptionally slowly, setting this parameter to a value less than one is advised.

#### Walker Placement Threshold (walkerThres)

The ```walkerThres``` parameter is similar to ```thres```, but for pixels over which walkers are placed. To save computational time and avoid edge artefacts, a ```walkerThres > thres``` value (like that set by the default) is recommended (e.g., use the default).


#### Walker Placement (walkers)

The ```walkers``` parameter allows the user to provide initial walker positions, rather than using the one defined by ```walkerThres```. The ```walkers``` can be used to continue a run with unconverged walkers, such as a test run with a lower number of maximum iterations (```maxT```).


#### Rescaling Axes (crdScaling)

When some of the axes in the data do not have well-established scaling relations with the others (e.g., velocity versues position), the rescaling parameter, ```crdScaling```, allows the user to rescale the axes relative to its native pixel values. This parameter can also be used to set the "effective" ```h``` value relative to each axis, allowing one to smooth each axis independently for a desired resolution.


#### Return Unconverged Walkers (return_unconverged)
If ```return_unconverged = True``` (i.e., the default), the run returns the converged walkers along with unconverged walkers. If desired, the unconverged walkers can be fed into a new run with the same data using the ```walkers``` parameter. This particular technique allows the users to continue a run that was terminated prematurely (e.g., in a quick test run with a low ```maxT``` or ```converge_frac```.
