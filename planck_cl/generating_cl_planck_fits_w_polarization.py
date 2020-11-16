#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pedro da Silveira Ferreira
"""
import numpy as np
import camb
import time
from camb import model, initialpower
from matplotlib import pyplot as plt
import healpy as hp

lmax = 2505
pars=camb.read_ini('planck_2018_camb_params.dat')
pars.set_for_lmax(lmax);
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)
dl_camb_py = powers['lensed_scalar'][:,0]
cl_camb_py = (2*np.pi*dl_camb_py[0:lmax+1]/(np.arange(lmax+1)*(np.arange(1,lmax+2))))/(10**12) # (10**12)->microK^2
cl_camb_py[0] = 0

calplanck = 1.000442**2

dl_planck_dat = np.loadtxt('COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt')

cl_planck_TT = ((2*np.pi*np.append(np.array([0,0]),dl_planck_dat[:,1])[0:lmax+1]/(np.arange(lmax+1)*(np.arange(1,lmax+2))))/(10**12))/calplanck
cl_planck_TT[0] = 0

cl_planck_TE = ((2*np.pi*np.append(np.array([0,0]),dl_planck_dat[:,2])[0:lmax+1]/(np.arange(lmax+1)*(np.arange(1,lmax+2))))/(10**12))/calplanck
cl_planck_TE[0] = 0

cl_planck_EE = ((2*np.pi*np.append(np.array([0,0]),dl_planck_dat[:,3])[0:lmax+1]/(np.arange(lmax+1)*(np.arange(1,lmax+2))))/(10**12))/calplanck
cl_planck_EE[0] = 0

plt.plot(np.arange(100,2001),(cl_camb_py[100:2001])/cl_planck_TT[100:2001])
plt.show()

plt.plot(np.arange(100,2001),cl_planck_TT[100:2001])
plt.plot(np.arange(100,2001),cl_planck_TE[100:2001])
plt.plot(np.arange(100,2001),cl_planck_EE[100:2001])
plt.show()

plt.plot(np.arange(100,2001),dl_planck_dat[:,1][100:2001])
plt.plot(np.arange(100,2001),dl_planck_dat[:,2][100:2001])
plt.plot(np.arange(100,2001),dl_planck_dat[:,3][100:2001])
plt.show()

#hp.write_cl('cl_TT_planck_2019.fits',cl_planck_TT,dtype=np.float64)
#hp.write_cl('cl_TE_planck_2019.fits',cl_planck_TE,dtype=np.float64)
#hp.write_cl('cl_EE_planck_2019.fits',cl_planck_EE,dtype=np.float64)
