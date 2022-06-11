# cmb-aber-dopp (version 0.2)

## Edit 14/02/22
The old wrong noise simulations pipeline was removed.

## Edit 14/04/21: 
* Addition of CCP alternative directly on harmonic space
* Addition of realistic and crossed expected error function with examples (forecasts).
* Small corrections on some comments and variable names

The cmb-aber-dopp repository include all basic code to reproduce some Planck systematics as the Dipole Distortion (DD), generate simulations of the Main Pipeline (MP) and Cross-check pipeline (CCP) (to generate aberrated, Dopplered and boosted simulations here is used the [Healpix Boost code](www.github.com/mquartin/healpix-boost)) and estimate the ![\beta^A
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EA%0A),![\beta^D
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED%0A) and ![\beta^B
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5EB%0A) observables on Planck data. For the DD case the derivation of the equations are included (needs [Mathematica](https://www.wolfram.com/mathematica/)). To cross-check Healpix Boost code (mainly when using polarization maps), DD cached maps and for fast tests, here it's included some python code for aberration and Doppler directly on pixel space. Some plots and source data for plots are included.

## Basic usage example of the python module
First, copy the cmbaberdopp.py (/python_module) module and cl_TT_planck_2019.fits (/planck_cl) files to same directory of your python code.

To generate a simple example we start with Doppler directly on the pixel space, with python. Let's start doing 32 CMB ideal simulations with ![\beta^D2
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED=0.00123%0A) pointing in the north direction, without aberration, and estimate the ![\beta^D
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Cbeta%5ED%0A) using ![\ell , \ell+1
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cell+%2C+%5Cell%2B1%0A) correlations.


```python
from cmbaberdopp import *
import healpy as hp
import numpy as np
from tqdm import tqdm

beta_var = 0.00123 # Planck CMB dipole
nside_var = 2048 # Planck nside
lmax_var = 2048
nsims = 32
latdir = 90 # 48.253  Dipole latitude
longdir = 0 # 264.021  Dipole longitude

cl_planck_TT = hp.read_cl('cl_TT_planck_2019.fits') # this file is present on /planck_cl folder
modulation_map = doppler_boost_map_dir_s(beta_var, nside_var, latdir, longdir) # modulation map on pixel space

#caching estimator theoretical terms
lmin_estim = 3
lmax_estim = 2002
binsize_estim = 10
htheofast(cl_planck_TT,lmin=lmin_estim,lmax=lmax_estim,binsize=binsize_estim)

# generating 32 simulations and estimating the betas (A,D,B)
n_threads = 4
for i in tqdm(range(nsims)):
    gaussianmap = hp.synfast(cl_planck_TT, nside_var, lmax=lmax_var,verbose=False)
    dopplered_map = gaussianmap*modulation_map
    doplered_alm = hp.map2alm(dopplered_map,iter=1) # iter=1 for fast test
    doplered_alm = reorder_idxpy2pix(doplered_alm,threads=n_threads) # changing from Healpy to Healpix fortran index order - betafast estimator only understand this ordering.
    betaabbins, betadoppbins, betaboostbins, betatotal, betatotalnorm = betafast(doplered_alm,lmin=lmin_estim,lmax=lmax_estim,binsize=binsize_estim,threads=n_threads,verbose=False,return_var=True) 
    # returns betaabbins, betadoppbins, betaboostbins, betatotal, betatotalnorm
    np.savetxt('betadopp_sim_'+str(i)+'.dat',betatotal[1]) 
    # 0 for Ab, 1 for Dopp and 2 for Boost - as we introduced only Dopp I'm getting only the final beta vector of Doppler estimator, others will be correlation that you can remove a posteriori.
    
beta_d_vector_list = [np.loadtxt('betadopp_sim_'+str(i)+'.dat')[2] for i in range(nsims)] # 2 is for z direction value (0 for x and 1 for y).
print('mean:',np.mean(beta_d_vector_list))
print('std:',np.std(beta_d_vector_list))

```

The functions htheofast and betafast are the implematation of the estimator introduced by [Amendola et al. (2011)](https://arxiv.org/abs/1008.1183). 
As the estimator is one of the most important features of this code this is why the example is focused on use it. Other important functions are the ones with the prefix "dipole_doppler_", related to the Dipole Distortion effect. 

## Dependencies
* python 3 (most of the code)
* fortran 90 (main boost code)
* Healpix
* [Healpix boost](https://github.com/mquartin/healpix-boost)
* Healpy (most of the code)
* numpy
* pymp (most of the code)
* astropy
* threadpoolctl
* tqdm
* [Mathematica](https://www.wolfram.com/mathematica/) (DD demonstration and part of analysis)

## References

* [Amendola et al. (2011)](https://arxiv.org/abs/1008.1183)

* [R. Catena and A. Notari (2012)](https://arxiv.org/abs/1210.2731).

* [A. Notari, M. Quartin and R. Catena (2014)](https://arxiv.org/abs/1304.3506).
