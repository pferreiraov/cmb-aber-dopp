# cmb-aber-dopp

The cmb-aber-dopp repository include all basic code to reproduce some Planck systematics as the Dipole Distortion (DD), simulate the Main Pipeline (MP) and Cros-check pipeline (CCP) (to generate aberrated, Doplered and boosted simulations here is used the Healpix Boost core - www.github.com/mquartin/healpix-boost) and estimate the $\beta^A$,$\beta^D$ and $\beta^B$ observables on Planck data. For the DD case the derivation of the equations are included. To cross-check Healpix Boost code (mainly when using polarization maps) and for fast tests, here it's included some python code for aberration and Doppler directly on pixel space.

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
