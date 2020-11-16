# cmb-aber-dopp (version 0.1)

The cmb-aber-dopp repository include all basic code to reproduce some Planck systematics as the Dipole Distortion (DD), simulate the Main Pipeline (MP) and Cros-check pipeline (CCP) (to generate aberrated, Doplered and boosted simulations here is used the Healpix Boost core - www.github.com/mquartin/healpix-boost) and estimate the ![\beta^A
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EA%0A),![\beta^D
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED%0A) and ![\beta^B
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EB%0A) observables on Planck data. For the DD case the derivation of the equations are included. To cross-check Healpix Boost code (mainly when using polarization maps), DD cached maps and for fast tests, here it's included some python code for aberration and Doppler directly on pixel space.

## Basic usage example of the python module
First, copy the cmbaberdopp_beta0p8.py module and cl_TT_planck_2019.fits files to same directory of your python code.

To generate a simple example we start with Doppler directly on the pixel space, with pyhton. Lets start doing 64 simulations with ![\beta^D2
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED=0.00123%0A) and no aberration.


```python
from cmbaberdopp_beta0p8.py import *
import healpy as hp
import numpy as np

beta_var = 0.00123 # Planck CMB dipole
nside_var = 2048 # Planck nside
lmax_var = 2048
latdir = 48.253 # Dipole latitude
longdir = 264.021 # Dipole longitude

cl_planck_TT = hp.read_cl('cl_TT_planck_2019.fits')
modulation_map = doppler_boost_map_dir_s(beta_var, nside_var, latdir, longdir) # modulation map on pixel space

#caching estimator theoretical terms
lmin_estim = 3
lmax_estim = 2002
binsize_estim = 10
htheofast(cl_planck_TT,lmin=lmin_estim,lmax=lmax_estim,binsize=binsize_estim)

# generating 64 simulations and estimating the betas (A,D,B)
for i in range(64):
  gaussianmap = hp.synfast(cl_planck_TT, nside_var, lmax=lmax_var)
  doplered_map = gaussianmap*modulation_map
  doplered_alm = hp.map2alm(doplered_map)
  doplered_alm = reorder_idxpy2pix(doplered_alm) # changing from Healpy to Healpix fortran index order - betafast estimator only understand this ordering.
  betaabbins, betadoppbins, betaboostbins, betatotal, betatotalnorm = betafast(doplered_alm,lmin=lmin_estim,lmax=lmax_estim,binsize=binsize_estim,return_var=True) 
  # returns betaabbins, betadoppbins, betaboostbins, betatotal, betatotalnorm
  np.savetxt('betadopp_sim'+str(i)+'.dat',betadoppbins)

```

The functions htheofast and betafast are the implematation of the estimator introduced by [Amendola et al. (2011)](https://arxiv.org/abs/1008.1183). 
As the estimator is one of the most important features of this code this is why the example is focused on use it. Other important functions are the "dipole_doppler_" related. 

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
