# cmb-aber-dopp

The cmb-aber-dopp repository include all basic code to reproduce some Planck systematics as the Dipole Distortion (DD), simulate the Main Pipeline (MP) and Cros-check pipeline (CCP) (to generate aberrated, Doplered and boosted simulations here is used the Healpix Boost core - www.github.com/mquartin/healpix-boost) and estimate the ![\beta^A
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EA%0A),![\beta^D
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED%0A) and ![\beta^B
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EB%0A) observables on Planck data. For the DD case the derivation of the equations are included. To cross-check Healpix Boost code (mainly when using polarization maps) and for fast tests, here it's included some python code for aberration and Doppler directly on pixel space.

## Usage example
First, copy the cmbboost_beta0p8.py module file to same directory of your python code.

To generate a simple example we start with Doppler directly on the pixel space, with pyhton. Lets start doing 64 simulations with ![\beta^D2
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED=0.00123%0A) and no aberration.


```python
import cmbboost_beta0p8
beta_var = 0.00123
nside_var = 2048
latdir = 48.253
longdir = 264.021
modulation_map = doppler_boost_map_dir_s(beta_var, nside_var, latdir, longdir)
...

```

## Dependencies
* python 3 (most of the code)
* fortran 90 (main boost code)
* Healpix
* Healpix boost
* Healpy (most of the code)
* numpy
* pymp
* astropy
* threadpoolctl
* tqdm
* Mathematica (DD demonstration and part of analysis)
