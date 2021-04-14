import healpy as hp
import numpy as np
from tqdm import tqdm
from astropy.io import fits as pyfits
import random
import pymp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import importlib
import sys
import multiprocessing as mp
from threadpoolctl import threadpool_limits
from math import sqrt
from math import radians
import gc
import subprocess

def c_nthreads_limit(n): # Function to change number of threads used on imported C/C++ functions
    threadpool_limits(limits=n, user_api='blas')
    threadpool_limits(limits=n, user_api='openmp')

#Generate a Doppler modulation pattern map. latdir and longdir set the boost direction on galactic coordinates (degrees)
def doppler_boost_map_dir(beta_var, nside_var, latdir, longdir,threads_var=mp.cpu_count()):
    npix = hp.nside2npix(nside_var)
    gamma = (1/(np.sqrt(1-beta_var*beta_var)))
    vecs = np.array(hp.pix2vec(nside_var,np.arange(hp.nside2npix(nside_var))))
    boostdirvec = hp.ang2vec(longdir,latdir,lonlat=True)
    boostedmap = pymp.shared.array((npix,), dtype='float64')    # Applying to a map.
    with pymp.Parallel(threads_var) as p:
        for pix in p.range(npix):
            boostedmap[pix] = (gamma*(1+(np.dot(vecs[:,pix],boostdirvec)*beta_var)))
    return boostedmap

#Alternative faster version    
#Generate a Doppler modulation pattern map. latdir and longdir set the boost direction on galactic coordinates (degrees)
def doppler_boost_map_dir_s(beta_var, nside_var, latdir, longdir):
    npix = hp.nside2npix(nside_var)
    gamma = (1/(np.sqrt(1-beta_var*beta_var)))
    vecs = np.array(hp.pix2vec(nside_var,np.arange(hp.nside2npix(nside_var)))).T
    boostdirvec = np.repeat(hp.ang2vec(longdir,latdir,lonlat=True).reshape(1,3),npix,axis=0)
    dotprocut = np.sum(vecs*boostdirvec,axis=1)*beta_var
    boostedmap = (gamma*(dotprocut+1))
    return boostedmap    

# Rotate a map w/out interpolation of overlay
def rotate_map(pixmap,latdir,longdir,threads=mp.cpu_count()):
    mapa=pixmap
    nside = hp.npix2nside(len(mapa)) # Setting the nside we will use
    npix = hp.nside2npix(nside) # Getting the number o pixels
    pixels = np.arange(npix) # [1] Array with the pixel indexes.
    theta_phi = hp.pix2ang(nside, pixels) # Getting pixel coordinates
    eulerlong=90+longdir
    eulerlat=90-latdir
    r = hp.Rotator(rot=[eulerlong,0,eulerlat])
    theta_rot, phi_rot = r(theta_phi[0], theta_phi[1])
    ipix_rot = hp.ang2pix(nside, theta_rot, phi_rot)
    maprotated = pymp.shared.array((npix,), dtype='float64')                 # Applying to a map.
    with pymp.Parallel(threads) as p:
        for pix in p.range(npix):
            maprotated[pix] = (mapa[ipix_rot[pix]])
    return maprotated

# Rotate a map w interpolation of overlay
def rotate_map_interpolated(pixmap,latdir,longdir):
    nside = hp.npix2nside(len(pixmap)) # Setting the nside we will use
    npix = hp.nside2npix(nside) # Getting the number o pixels
    pixels = np.arange(npix) # [1] Array with the pixel indexes.
    theta_phi = hp.pix2ang(nside, pixels) # Getting pixel coordinates
    eulerlong=90-longdir
    eulerlat=90-latdir
    r = hp.Rotator(rot=[0,0,eulerlat])
    maprotated = r.rotate_map_pixel(pixmap)
    r = hp.Rotator(rot=[eulerlong,0,0])
    maprotated = r.rotate_map_pixel(maprotated)
    return maprotated

# Reorder a_lm from Healpix index order to Healpy index order #
# Attention: When you import fits file from Healpix, if hits file have the a_lms indexes stored, Healpy will reorder it
def reorder_idxpix2py(alm_pix,threads=mp.cpu_count()):
    lmax = hp.Alm.getlmax(len(alm_pix))
    idxpix2py=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='int64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1)/2
            for m in range(0,l+1):
                idxpix2py[int(hp.Alm.getidx(lmax,l,m))]=int(m+idxl)
    return alm_pix[idxpix2py]

# Generate a_lm index array to reorder from Healpix index order to Healpy index order #
# Attention: When you import fits file from Healpix, if hits file have the a_lms indexes stored, Healpy will reorder it #
def reorder_idx_idxpix2py(lmax,threads=mp.cpu_count()):
    idxpix2py=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='int64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1)/2
            for m in range(0,l+1):
                idxpix2py[int(hp.Alm.getidx(lmax,l,m))]=int(m+idxl)
    return idxpix2py

# Reorder a_lm from Healpy index order to Healpix index order 
def reorder_idxpy2pix(alm_py,threads=mp.cpu_count()):
    lmax = hp.Alm.getlmax(len(alm_py))
    idxpy2pix=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='int64') # NOW ITS PY2PIX!
    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1)/2
            for m in range(0,l+1):
                idxpy2pix[int(m+idxl)]=int(hp.Alm.getidx(lmax,l,m))
    return alm_py[idxpy2pix]

# Generate a_lm index array to reorder from Healpy index order to Healpix index order #
def reorder_idx_idxpy2pix(lmax,threads=mp.cpu_count()):
    idxpy2pix=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='int64') # NOW ITS PY2PIX!
    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1)/2
            for m in range(0,l+1):
                idxpy2pix[int(m+idxl)]=int(hp.Alm.getidx(lmax,l,m))
    return idxpy2pix

# Slice an a_lm in bins of some angular scale range 
# ex.: for binsize=100 -> bins: (a_lm from) 0<=l<=99, 100<=l<=199, 200<=l<=299, ... , lmax-100<=l<=lmax
# overbin parameter create bins intersections, this is important for Doppler and Aberration correlation, 
# because they are not orthogonal, so there is signal leakage between bins.
# ex.: for binsize=100 -> bins: (a_lm from) 0<=l<=99+overbin, 100-overbin<=l<=199+overbin, ... , lmax-100-overbin<=l<=lmax
def alm_filters(lmax,binsize,overbin,threads=mp.cpu_count()):
    nslices = int(lmax/binsize) # Attention! 
    filtertable=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            if n==0:
                o1=np.ones(hp.Alm.getsize(lmax=(binsize-1+overbin)))
                z1=np.zeros(hp.Alm.getsize(lmax=lmax)-(hp.Alm.getsize(lmax=(binsize-1+overbin))))
                filtertable[n]=(np.append(o1,z1))
            if n>0 and n<int((nslices-1)):
                z1=np.zeros(hp.Alm.getsize(lmax=(binsize*n)-1-overbin))
                o1=np.ones(hp.Alm.getsize(lmax=(binsize*(n+1))-1+overbin)-hp.Alm.getsize(lmax=(binsize*n)-1-overbin))
                z2=np.zeros(hp.Alm.getsize(lmax=lmax)-hp.Alm.getsize(lmax=(binsize*(n+1))-1+overbin))
                filtertable[n]=(np.append(z1,np.append(o1,z2)))
            if n==int((nslices-1)):
                z1=np.zeros(hp.Alm.getsize(lmax=(binsize*n)-1-overbin))
                o1=np.ones(hp.Alm.getsize(lmax=lmax)-hp.Alm.getsize(lmax=(binsize*n)-1-overbin))
                filtertable[n]=(np.append(z1,o1))
    return filtertable 

################################################################# 
def cps0factor( l, m):                                         ## ArXiv XXXXXXXXXX
    return sqrt(((l+1+m)*(l+1-m))/((4*(l+1)*(l+1)-1)))         ## Equations XXXXXXXX* (for s=0 and s=1)
                                                               ## *(considering symmetry on x,y)
def cms0factor( l, m):                                         ##
    return -1*sqrt(((l+m)*(l-m))/(4*l*l-1))                    ##
                                                               ##
def cps1factor( l, m):                                         ##
    return -1*sqrt(((l+2+m)*(l+m+1))/((4*(l+1)*(l+1)-1)*(2)))  ##
                                                               ##
def cms1factor( l, m):                                         ##
    return -1*sqrt(((l+m-1)*(l+m))/((4*l*l-1)*(2)))            ##
#################################################################

#################################################################
def almf(almlist_array, l, m ):                                ## ArXiv XXXXXXXXXX
    ellement=almlist_array[(l*(l+1))+m]                        ## Equations XXXXXXXX
    return ellement;                                           ##
                                                               ##
def almfconjugate(almlistconjugate_array, l, m ):              ##
    ellement=almlistconjugate_array[(l*(l+1))+m]               ##
    return ellement;                                           ##
#################################################################

# htheofast pre-computes every theoretical variable of the beta estimator (ArXiv XXXXXXXXXX, Equations XXXXXXXX)
# and save then on global variables, so betafast function can use it.
#
# Parameters: ############################################################################################
#                                                                                                       ##
# cl -> the array of Cl (cmb angular power spectrum)                                                    ##
# clfile -> path to Cl file if you dont input the array                                                 ##
# nlfile -> noise angular power spectrum                                                                ##
# mastermatrixfile -> master matrix for mask effect on Cls                                              ##
# arcminbeam -> Gaussian beam (arcmin)                                                                  ##
# lmin and lmax -> minimun and maximum l value                                                          ##
# binsize -> binsize that you will use on beta estimator                                                ##
# sufixname -> sufix to export files w/ you want to save theoretical values                             ##
# threads -> number of threads you want to use (default: All)                                           ##
# plot_std -> plot expected standard deviation for aberration, doppler and Aberration+Doppler estimator ##
##########################################################################################################

def htheofast(cl,clfile=False,nlfile=False,mastermatrixfile=False,arcminbeam=0,lmin=3,lmax=2002,binsize=10,sufixname='test',threads=mp.cpu_count(),plot_std=False,lmin_plot=1400,lmax_plot=2000,export_fg=False,export_sumbottom=False):
   
    start = time.time()
    lmax=int(lmax+1)
    lmaxalm=int(lmax+1)
    if mastermatrixfile != False:
        mastermatrix=np.genfromtxt(mastermatrixfile, delimiter=',')
        mastermatrix[:,0:2]=0
        mastermatrizsize=len(mastermatrix)
    ########## IMPORTING DATA ##########
    if clfile==True:
        cl=np.array((pyfits.open(cl))[1].data['TEMPERATURE'],dtype='float64')
    if nlfile != False:
        nl=hp.read_cl(nlfile)
        if mastermatrixfile != False:
            cl=cl[:mastermatrizsize]
            cl=np.dot(mastermatrix,cl)[0:lmax+2]+nl[0:lmax+2] ## ATTENTION! Depends on n_l inputed (I'm using a masked n_l)
        else:
            nl=nl[:lmax+10]
            cl=cl[:lmax+10]+nl
    if nlfile == False:
        if mastermatrixfile != False:
            cl=cl[:mastermatrizsize]
            cl=np.dot(mastermatrix,cl)  
    if arcminbeam != 0:   
        gaussianbeam=hp.gauss_beam(radians(arcminbeam/60),lmax=lmax)
        gaussianbeam2=gaussianbeam*gaussianbeam
        cl=cl/gaussianbeam2
    global cl_conv
    cl_conv = cl

    ########## CALCULATING c+ AND c- ##########
    cps0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmaxalm+1):
            idxl=l*(l+1)
            for m in range(-l,l+1):
                idxlm=idxl+m
                cps0=cps0factor( l, m)
                cms0=cms0factor( l, m)
                cps1=cps1factor( l, m)
                cms1=cms1factor( l, m)
                cps0abfactorlist[idxlm]=(l+2)*cps0
                cms0abfactorlist[idxlm]=(l-1)*cms0
                cps1abfactorlist[idxlm]=(l+2)*cps1
                cms1abfactorlist[idxlm]=(l-1)*cms1
                cps0doppfactorlist[idxlm]=-cps0
                cms0doppfactorlist[idxlm]=cms0
                cps1doppfactorlist[idxlm]=-cps1
                cms1doppfactorlist[idxlm]=cms1
    cms0abdoppfactorlist=cms0doppfactorlist+cms0abfactorlist
    cms1abdoppfactorlist=cms1doppfactorlist+cms1abfactorlist
    cps0abdoppfactorlist=cps0doppfactorlist+cps0abfactorlist
    cps1abdoppfactorlist=cps1doppfactorlist+cps1abfactorlist

    ########## CALCULATING <f> AND <g> (h) ##########
    symarraysize=hp.Alm.getsize(lmax=lmax)*2-lmax-2
    notsymarraysize=hp.Alm.getsize(lmax=lmax)
    
    global f0ab_theo, f0dopp_theo, f0abdopp_theo
    global f02ab_theo, f02dopp_theo, f02abdopp_theo
    global f1ab_theo, f1dopp_theo, f1abdopp_theo
    global g1ab_theo, g1dopp_theo, g1abdopp_theo
    f0ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f1ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')

    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1);idxlp1=(l+1)*(l+2);idxld2=int(idxl/2);lp1=l+1;
            for m in range(1,l):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl[l]+cps0abfactorlist[idxlm]*cl[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl[l]+cps0doppfactorlist[idxlm]*cl[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl[l]-cps1abfactorlist[idxlm]*cl[lp1])
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl[l]-cps1doppfactorlist[idxlm]*cl[lp1])
            for m in range(-l+1,0):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl[l]-cps1abfactorlist[idxlm]*cl[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl[l]-cps1doppfactorlist[idxlm]*cl[lp1])
            for m in [l]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl[l]+cps0abfactorlist[idxlm]*cl[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl[l]+cps0doppfactorlist[idxlm]*cl[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
            for m in [0]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f0ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl[l]+cps0abfactorlist[idxlm]*cl[lp1])
                f02ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl[l]+cps0abfactorlist[idxlm]*cl[lp1])
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl[l]-cps1abfactorlist[idxlm]*cl[lp1])
                f0dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl[l]+cps0doppfactorlist[idxlm]*cl[lp1])
                f02dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl[l]+cps0doppfactorlist[idxlm]*cl[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl[l]-cps1doppfactorlist[idxlm]*cl[lp1])

    f0abdopp_theo=f0ab_theo+f0dopp_theo
    f1abdopp_theo=f1ab_theo+f1dopp_theo
    f02abdopp_theo=f02ab_theo+f02dopp_theo

    sqrt2m=1/sqrt(2)
    f1ab_theo=sqrt2m*f1ab_theo
    f1dopp_theo=sqrt2m*f1dopp_theo
    f1abdopp_theo=sqrt2m*f1abdopp_theo
    g1ab_theo=f1ab_theo
    g1dopp_theo=f1dopp_theo
    g1abdopp_theo=f1abdopp_theo
    
    #Dealocating memory
    cps0abfactorlist=None;cms0abfactorlist=None;cps1abfactorlist=None;cms1abfactorlist=None;
    cps0doppfactorlist=None;cms0doppfactorlist=None;cps1doppfactorlist=None;cms1doppfactorlist=None;
    cms0abdoppfactorlist=None;cms1abdoppfactorlist=None;cps0abdoppfactorlist=None;cps1abdoppfactorlist=None;

    ########## CALCULATING Cl*Cl+1 ##########
    global clclplus1
    clclplus1=cl[1:][0:lmax+2]*cl[:-1][0:lmax+2]

    ########## CALCULATING bottom sum ##########
    global sumbottomzab, sumbottomxab, sumbottomyab
    global sumbottomzdopp, sumbottomxdopp, sumbottomydopp
    global sumbottomzabdopp, sumbottomxabdopp, sumbottomyabdopp
    sumbottomzab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomydopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmin,lmax):
            idxl=int(l*(l+1));idxld2=int(idxl/2);
            sumbottomzab[l-lmin]=np.sum((f02ab_theo[idxld2:idxld2+l+1]*f0ab_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxab[l-lmin]=np.sum((f1ab_theo[idxl-l:idxl+l+1]*f1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomyab[l-lmin]=np.sum((g1ab_theo[idxl-l:idxl+l+1]*g1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomzdopp[l-lmin]=np.sum((f02dopp_theo[idxld2:idxld2+l+1]*f0dopp_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxdopp[l-lmin]=np.sum((f1dopp_theo[idxl-l:idxl+l+1]*f1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomydopp[l-lmin]=np.sum((g1dopp_theo[idxl-l:idxl+l+1]*g1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomzabdopp[l-lmin]=np.sum((f02abdopp_theo[idxld2:idxld2+l+1]*f0abdopp_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxabdopp[l-lmin]=np.sum((f1abdopp_theo[idxl-l:idxl+l+1]*f1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomyabdopp[l-lmin]=np.sum((g1abdopp_theo[idxl-l:idxl+l+1]*g1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))

    sumbottomab=np.array([sumbottomxab,sumbottomyab,sumbottomzab])
    sumbottomdopp=np.array([sumbottomxdopp,sumbottomydopp,sumbottomzdopp])
    sumbottomabdopp=np.array([sumbottomxabdopp,sumbottomyabdopp,sumbottomzabdopp])
    
    global inversevarianceabxbin, inversevarianceabybin, inversevarianceabzbin
    global inversevariancedoppxbin, inversevariancedoppybin, inversevariancedoppzbin
    global inversevarianceabdoppxbin, inversevarianceabdoppybin, inversevarianceabdoppzbin
    inversevarianceabxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')

    with pymp.Parallel(threads) as p:
        for i in p.range(int((lmax-lmin)/binsize)):
            inversevarianceabxbin[i]=(np.sum(sumbottomxab[i*binsize:(i+1)*binsize]))
            inversevarianceabybin[i]=(np.sum(sumbottomyab[i*binsize:(i+1)*binsize]))
            inversevarianceabzbin[i]=(np.sum(sumbottomzab[i*binsize:(i+1)*binsize]))
            inversevariancedoppxbin[i]=(np.sum(sumbottomxdopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppybin[i]=(np.sum(sumbottomydopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppzbin[i]=(np.sum(sumbottomzdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppxbin[i]=(np.sum(sumbottomxabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppybin[i]=(np.sum(sumbottomyabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppzbin[i]=(np.sum(sumbottomzabdopp[i*binsize:(i+1)*binsize]))

    if export_fg == True:
        np.savetxt(sufixname+'_'+'f02ab_theo.dat',f02ab_theo)
        np.savetxt(sufixname+'_'+'f1ab_theo.dat',f1ab_theo)
        np.savetxt(sufixname+'_'+'g1ab_theo.dat',g1ab_theo)
        np.savetxt(sufixname+'_'+'f02dopp_theo.dat',f02dopp_theo)
        np.savetxt(sufixname+'_'+'f1dopp_theo.dat',f1dopp_theo)
        np.savetxt(sufixname+'_'+'g1dopp_theo.dat',g1dopp_theo)
        np.savetxt(sufixname+'_'+'f02abdopp_theo.dat',f02abdopp_theo)
        np.savetxt(sufixname+'_'+'f1abdopp_theo.dat',f1abdopp_theo)
        np.savetxt(sufixname+'_'+'g1abdopp_theo.dat',g1abdopp_theo)

    if export_sumbottom == True:    
        np.savetxt(sufixname+'_'+'sumbottom_ab_sim_'+str(n+1)+'.dat',sumbottomab)
        np.savetxt(sufixname+'_'+'sumbottom_dopp_sim_'+str(n+1)+'.dat',sumbottomxdopp)
        np.savetxt(sufixname+'_'+'sumbottom_abdopp_sim_'+str(n+1)+'.dat',sumbottomabdopp)
    
    global stdabz_cumsum, stddoppz_cumsum, stdabdoppz_cumsum
    c=299792.458
    if plot_std == True:
        stdabz_cumsum=np.sqrt(1/(np.cumsum(sumbottomzab)))*c
        stddoppz_cumsum=np.sqrt(1/(np.cumsum(sumbottomzdopp)))*c
        stdabdoppz_cumsum=np.sqrt(1/(np.cumsum(sumbottomzabdopp)))*c
        plt.plot(np.arange(lmin_plot,lmax_plot),stdabz_cumsum[lmin_plot-1:lmax_plot-1], label='Ab')
        plt.plot(np.arange(lmin_plot,lmax_plot),stddoppz_cumsum[lmin_plot-1:lmax_plot-1], label='Dopp')
        plt.plot(np.arange(lmin_plot,lmax_plot),stdabdoppz_cumsum[lmin_plot-1:lmax_plot-1], label='AbDopp')
        plt.title('Theoretical Standard Deviation')
        plt.ylabel('\u03C3 (km/s)')
        plt.xlabel('\u2113')
        plt.legend()
        plt.grid()
        plt.show()
    
    end = time.time()
    print('Completed in: ',end-start,' seconds.')

# betafast computes the beta estimator* (ArXiv XXXXXXXXXX, Equations XXXXXXXX)
# *You have to run htheofast before it.
#
# Parameters: #################################################################################################
#                                                                                                            ##
# alm_var -> the array of a_lm (cmb spherical harmonics components) ## Have to be on Healpix ordering!       ##
# almfile -> path to a_lm file if you dont input the array          ## Have to be on Healpix ordering!       ##
# lmin and lmax -> minimun and maximum l value                                                               ##
# binsize -> binsize that you will use on beta estimator. (set the range of sums on l)                       ##
# sufixname -> sufix to export files w/ you want to save the values (you have to set exports_results=True)   ##
# threads -> number of threads you want to use (default: All)                                                ##
# return_var -> If True returns beta values, else (default) only export/save values (if exports_results=True)##
###############################################################################################################

def betafast(alm_var,almfile=False,lmin=3,lmax=2002,binsize=10,sufixname='test',threads=mp.cpu_count(),export_results=False,verbose=True,return_var=False):
    start = time.time()
    lmax=int(lmax+1)
    lmaxalm=int(lmax+1)
    if 'clclplus1' in globals() and 'sumbottomxab' in globals():
        pass
    else:
        print('clclplus1 and sumbottom\'s variables don\'t exists, did you run the htheo_fast function before?')
        return

    ########## IMPORTING DATA ##########
    if almfile==True:
        alm=pyfits.open(alm_var)
        almreal=np.array(alm[1].data['REAL'])
        almimag=np.array(alm[1].data['IMAG'])
    else:
        almreal=alm_var.real
        almimag=alm_var.imag

    ########## GENERATING FULL a_lms (w/ negative "m" values) ##########
    almlistreal= pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    almlistimag= pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmaxalm+1):
            idxl=l*(l+1);idxld2=int(idxl/2);
            for m in range(0,l+1):
                idxm=m
                almlistreal[idxl+idxm]=(almreal[idxld2+idxm])
                almlistimag[idxl+idxm]=(almimag[idxld2+idxm])
            for m in range(-l,0):
                idxm=m;idxmminus=-m;symfactor=((-1)**m);
                almlistreal[idxl+idxm]=((symfactor)*(almreal[idxld2+idxmminus]))
                almlistimag[idxl+idxm]=-1*(symfactor)*(almimag[idxld2+idxmminus])

    almlist=almlistreal+almlistimag*1j
    almlistconjugate=almlistreal-almlistimag*1j
    almlistreal=None;almlistimag=None;almreal=None;almimag=None;alm=None;

    ########## CALCULATING f^obs and g^obs ##########
    f0_obs=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f1_obs=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    g1_obs=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1);idxl2=int(idxl/2);lp1=l+1;
            for m in range(0,l):
                f0_obs[idxl2+m]=((almfconjugate(almlistconjugate,l,m)*almf(almlist,lp1,m)).real)
                correlation1=(almfconjugate(almlistconjugate,l,m)*almf(almlist,lp1,m+1))
                f1_obs[idxl+m]=(correlation1.real)
                g1_obs[idxl+m]=(correlation1.imag)
            for m in range(-l+1,0):
                correlation1=(almfconjugate(almlistconjugate,l,m)*almf(almlist,lp1,m+1))
                f1_obs[idxl+m]=(correlation1.real)
                g1_obs[idxl+m]=(correlation1.imag)
            for m in [l]:
                f0_obs[idxl2+m]=((almfconjugate(almlistconjugate,l,m)*almf(almlist,lp1,m)).real)

    f0_obs=-f0_obs
    almlist=None;almlistconjugate=None;

    ########## CALCULATING top sum ##########
    sumtopzab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopxab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopyab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopzdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopxdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopydopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopzabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopxabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumtopyabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmin,lmax):
            idxl=int(l*(l+1));idxl2=int(idxl/2);
            sumtopzab[l-lmin]=np.sum((f0_obs[idxl2:idxl2+l+1]*f02ab_theo[idxl2:idxl2+l+1]/clclplus1[l]))
            sumtopxab[l-lmin]=np.sum((f1_obs[idxl-l:idxl+l+1]*f1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumtopyab[l-lmin]=np.sum((g1_obs[idxl-l:idxl+l+1]*g1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumtopzdopp[l-lmin]=np.sum((f0_obs[idxl2:idxl2+l+1]*f02dopp_theo[idxl2:idxl2+l+1]/clclplus1[l]))
            sumtopxdopp[l-lmin]=np.sum((f1_obs[idxl-l:idxl+l+1]*f1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumtopydopp[l-lmin]=np.sum((g1_obs[idxl-l:idxl+l+1]*g1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumtopzabdopp[l-lmin]=np.sum((f0_obs[idxl2:idxl2+l+1]*f02abdopp_theo[idxl2:idxl2+l+1]/clclplus1[l]))
            sumtopxabdopp[l-lmin]=np.sum((f1_obs[idxl-l:idxl+l+1]*f1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumtopyabdopp[l-lmin]=np.sum((g1_obs[idxl-l:idxl+l+1]*g1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))

    ########## CALCULATING beta bins and beta total ##########
    betaxabbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betayabbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betazabbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betaxdoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betaydoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betazdoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betaxabdoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betayabdoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    betazabdoppbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')

    with pymp.Parallel(threads) as p:
        for i in p.range(int((lmax-lmin)/binsize)):
            betaxabbin[i]=(np.sum(sumtopxab[i*binsize:(i+1)*binsize])/inversevarianceabxbin[i])
            betayabbin[i]=-(np.sum(sumtopyab[i*binsize:(i+1)*binsize])/inversevarianceabybin[i])
            betazabbin[i]=(np.sum(sumtopzab[i*binsize:(i+1)*binsize])/inversevarianceabzbin[i])
            betaxdoppbin[i]=(np.sum(sumtopxdopp[i*binsize:(i+1)*binsize])/inversevariancedoppxbin[i])
            betaydoppbin[i]=-(np.sum(sumtopydopp[i*binsize:(i+1)*binsize])/inversevariancedoppybin[i])
            betazdoppbin[i]=(np.sum(sumtopzdopp[i*binsize:(i+1)*binsize])/inversevariancedoppzbin[i])
            betaxabdoppbin[i]=(np.sum(sumtopxabdopp[i*binsize:(i+1)*binsize])/inversevarianceabdoppxbin[i])
            betayabdoppbin[i]=-(np.sum(sumtopyabdopp[i*binsize:(i+1)*binsize])/inversevarianceabdoppybin[i])
            betazabdoppbin[i]=(np.sum(sumtopzabdopp[i*binsize:(i+1)*binsize])/inversevarianceabdoppzbin[i])


########## ATTENTION GALACTIC COORDINATES! ##########
########## Generating and exporting results ##########
    betaxabtotal=np.sum(betaxabbin*inversevarianceabxbin)/(np.sum(inversevarianceabxbin))
    betayabtotal=np.sum(betayabbin*inversevarianceabybin)/(np.sum(inversevarianceabybin))
    betazabtotal=np.sum(betazabbin*inversevarianceabzbin)/(np.sum(inversevarianceabzbin))
    betaabbins=np.array([betaxabbin,betayabbin,betazabbin])

    betaxdopptotal=np.sum(betaxdoppbin*inversevariancedoppxbin)/(np.sum(inversevariancedoppxbin))
    betaydopptotal=np.sum(betaydoppbin*inversevariancedoppybin)/(np.sum(inversevariancedoppybin))
    betazdopptotal=np.sum(betazdoppbin*inversevariancedoppzbin)/(np.sum(inversevariancedoppzbin))
    betadoppbins=np.array([betaxdoppbin,betaydoppbin,betazdoppbin])

    betaxabdopptotal=np.sum(betaxabdoppbin*inversevarianceabdoppxbin)/(np.sum(inversevarianceabdoppxbin))
    betayabdopptotal=np.sum(betayabdoppbin*inversevarianceabdoppybin)/(np.sum(inversevarianceabdoppybin))
    betazabdopptotal=np.sum(betazabdoppbin*inversevarianceabdoppzbin)/(np.sum(inversevarianceabdoppzbin))
    betaabdoppbins=np.array([betaxabdoppbin,betayabdoppbin,betazabdoppbin])

    betaabtotal=np.array([betaxabtotal,betayabtotal,betazabtotal])
    betadopptotal=np.array([betaxdopptotal,betaydopptotal,betazdopptotal])
    betaabdopptotal=np.array([betaxabdopptotal,betayabdopptotal,betazabdopptotal])
    betatotal=np.array([betaabtotal,betadopptotal,betaabdopptotal])
    betatotalnorm=np.array([np.linalg.norm(betaabtotal),np.linalg.norm(betadopptotal),np.linalg.norm(betaabdopptotal)])

    if export_results == True:
#        np.savetxt(sufixname+'_'+'sumtop_ab.dat',sumtopab)
#        np.savetxt(sufixname+'_'+'sumtop_dopp_sim.dat',sumtopdopp)
#        np.savetxt(sufixname+'_'+'sumtop_abdopp_sim.dat',sumtopabdopp)
        np.savetxt(sufixname+'_'+'beta_ab_bins_size'+str(binsize)+'.dat',betaabbins)
        np.savetxt(sufixname+'_'+'beta_dopp_bins_size'+str(binsize)+'.dat',betadoppbins)
        np.savetxt(sufixname+'_'+'beta_abdopp_bins_size'+str(binsize)+'.dat',betaabdoppbins)
#        np.savetxt(sufixname+'_'+'betatotal.dat',betatotal)
#        np.savetxt(sufixname+'_'+'betatotal_norm.dat',betatotalnorm)
    if verbose==True:
        end = time.time()
        print('Completed in: ',end-start,' seconds.')
        print('Attention: Results on cartesian galactic coordinates (left handed cartesian coordinates)!')
        print('Attention: The C_l used on htheofast function have to be the same used to generated the a_lm\'s and with the same [lmin,lmax] range!')

    if return_var==True:
        return betaabbins, betadoppbins, betaabdoppbins, betatotal, betatotalnorm

# Load dipole_doppler modulation maps (these maps have to be created before!)  
# binsize = 200 because is the precomputated "binned maps" I generated before, up to lmax=2100 w/ nside=2048 
# (you can change it if you generate other the Dipole Doppler maps before).
# low_res suffix is due to the "low angular scale resulution" -> bin size
# ATTENTION: VERY DEPENDENT OF PRE-COMPUTED DIPOLE DOPPLER MAPS!
# Parameters: #################################################################################################
#                                                                                                            ##
# alm -> the array of a_lm (cmb spherical harmonics components                                               ##
# pipeline -> depends on precomputed maps, for this work we have "smica" and "nilc"                          ##
# almfile -> path to a_lm file if you dont input the array                                                   ##
# lmax -> minimun and maximum l value                                                                        ##
# binsize -> binsize that you will use on beta estimator. (set the range of sums on l)                       ##
# threads -> number of threads you want to use (default: All)                                                ##
###############################################################################################################
def dipole_doppler_load_low_res(alm,pipeline,dd_dir,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=mp.cpu_count(),verbose=True):

    start = time.time()
    
    nslices = int(lmax/binsize)
    
    if lmax > 2100:
        print('Sorry, no precomputed Doppler map for dipole Doppler over l=2100. Please, compute these maps and put on the doppler maps folder.')
        return
    
    ########## CHECKING a__lms LMAX #########
    if almfile == True:
        almoriginal=hp.read_alm(str(alm))
        lsize=hp.Alm.getlmax(len(almoriginal))
    if almfile == False:
        lsize=hp.Alm.getlmax(len(alm))
        
    ########## IMPORTING DATA ########## 
    global dipole_doppler_map
    dipole_doppler_map = np.zeros((nslices,hp.nside2npix(nside)))
    for i in tqdm(range(nslices),desc='Loading '+str(pipeline)+' dipole Doppler precomputed maps.'):
        dipole_doppler_map[i] = hp.read_map(str(dd_dir)+'low_res_'+str(pipeline).lower()+'_dipole_doppler_bin'+str(i)+'.fits',verbose=False)
    
    ########## INDEX CONVERSION TABLES ##########
    print('The a_lm file is used to check the expected file structure. Remeber to use Healpy index order on a_lm\'s.')
    print('Calculating py2pix and pix2py index conversion tables')
    
    global idxpix2py, idxpy2pix, idxpy2pixlmax, idxpy2pixslices, idxpy2pixhalflmax
    idxpix2py = reorder_idx_idxpix2py(lmax,threads)
    idxpy2pix = reorder_idx_idxpy2pix(lsize,threads)
    idxpy2pixlmax = reorder_idx_idxpy2pix(lmax,threads)
    idxpy2pixhalflmax = reorder_idx_idxpy2pix(int(lmax/2),threads)
    
    print('Tables generated.')

    ########## GENERATING FILTER TABLES ##########
    print('Generating filter tables (harmonic space filter).')
    
    global filtertable, filtertablemerge
    filtertable = alm_filters(lmax,binsize,overbin,threads)
    filtertablemerge = alm_filters(lmax,binsize,0,threads)
    
    print('Filters generated.')
    end = time.time()
    print('Completed in: ',end-start,' seconds.')
    
    if verbose == True:
        print('Dipole doppler preamble loaded.')
        print('\u0394 \u2113 = '+str(binsize))

# dipole_doppler_low_res apply the Dipole Doppler effect to an a_lm (input and output on healpy ordering!) 
# You need load dipole_doppler modulation maps before (using dipole_doppler_load_low_res)
# Parameters: #################################################################################################
#                                                                                                            ##
# alm -> the array of a_lm (cmb spherical harmonics components ## Healpy ordering                            ##
# almfile -> path to a_lm file if you dont input the array     ## Healpy ordering                            ##
# lmax -> minimun and maximum l value                                                                        ##
# binsize -> binsize that you will use on beta estimator. (set the range of sums on l)                       ##
# threads -> number of threads you want to use (default: All)                                                ##
###############################################################################################################

def dipole_doppler_load_low_res_pol(alm,pipeline,dd_dir,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=mp.cpu_count(),verbose=True):

    start = time.time()
    
    nslices = int(lmax/binsize)
    
    if lmax > 2100:
        print('Sorry, no precomputed Doppler map for dipole Doppler over l=2100. Please, compute these maps and put on the doppler maps folder.')
        return
    
    ########## CHECKING a__lms LMAX #########
    if almfile == True:
        almoriginal=hp.read_alm(str(alm))
        lsize=hp.Alm.getlmax(len(almoriginal))
    if almfile == False:
        lsize=hp.Alm.getlmax(len(alm))
        
    ########## IMPORTING DATA ########## 
    global dipole_doppler_map_pol
    dipole_doppler_map_pol = np.zeros((nslices,hp.nside2npix(nside)))
    for i in tqdm(range(nslices),desc='Loading '+str(pipeline)+' dipole Doppler precomputed maps.'):
        dipole_doppler_map_pol[i] = hp.read_map(str(dd_dir)+'low_res_'+str(pipeline).lower()+'_dipole_doppler_pol_bin'+str(i)+'.fits',verbose=False)
    
    ########## INDEX CONVERSION TABLES ##########
    print('The a_lm file is used to check the expected file structure. Remeber to use Healpy index order on a_lm\'s.')
    print('Calculating py2pix and pix2py index conversion tables')
    
    global idxpix2py, idxpy2pix, idxpy2pixlmax, idxpy2pixslices, idxpy2pixhalflmax
    idxpix2py = reorder_idx_idxpix2py(lmax,threads)
    idxpy2pix = reorder_idx_idxpy2pix(lsize,threads)
    idxpy2pixlmax = reorder_idx_idxpy2pix(lmax,threads)
    idxpy2pixhalflmax = reorder_idx_idxpy2pix(int(lmax/2),threads)
    
    print('Tables generated.')

    ########## GENERATING FILTER TABLES ##########
    print('Generating filter tables (harmonic space filter).')
    
    global filtertable, filtertablemerge
    filtertable = alm_filters(lmax,binsize,overbin,threads)
    filtertablemerge = alm_filters(lmax,binsize,0,threads)
    
    print('Filters generated.')
    end = time.time()
    print('Completed in: ',end-start,' seconds.')
    
    if verbose == True:
        print('Dipole doppler preamble loaded.')
        print('\u0394 \u2113 = '+str(binsize))

# dipole_doppler_low_res apply the Dipole Doppler effect to an a_lm (input and output on healpy ordering!) 
# You need load dipole_doppler modulation maps before (using dipole_doppler_load_low_res)
# Parameters: #################################################################################################
#                                                                                                            ##
# alm -> the array of a_lm (cmb spherical harmonics components ## Healpy ordering                            ##
# almfile -> path to a_lm file if you dont input the array     ## Healpy ordering                            ##
# lmax -> minimun and maximum l value                                                                        ##
# binsize -> binsize that you will use on beta estimator. (set the range of sums on l)                       ##
# threads -> number of threads you want to use (default: All)                                                ##
###############################################################################################################

def dipole_doppler_low_res(alm,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=int(mp.cpu_count()/4),processes=int(4)):
    
    start_time = time.time()
    
    nslices = int(lmax/binsize)
    
    if 'dipole_doppler_map' in globals():
        pass
    else:
        print('dipole doppler maps are not loaded yet, did you run the dipole_doppler_load function before?')
        return
    if almfile == True:
        almoriginal=hp.read_alm(alm)
    else:
        almoriginal = alm
    
    almoriginal=almoriginal[idxpy2pix]
    almoriginal=almoriginal[:hp.Alm.getsize(lmax)]

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=almoriginal.real*filtertable[n]
            almslices_imag[n]=almoriginal.imag*filtertable[n]
    almsliced = almslices_real + almslices_imag*1j
    almsliced = list(almsliced)
    
    print('Running alm2map on each slice and applying dipole Doppler')
    global alm2map_slices
    def alm2map_slices(i):
        almslicedpy=almsliced[i][idxpix2py]
        result = hp.alm2map(almslicedpy,nside=nside,lmax=lmax,verbose=False)
        result = np.multiply(result,dipole_doppler_map[i])
        almsliced[i] = None # Clear memory (preliminary results)
        return result
    pool = mp.Pool(processes=processes)
    binnedmaps = pool.map(alm2map_slices, range(nslices))
    pool.terminate()
    pool.join()
    alm2map_slices = None #Clear memory (preliminary results)

    print('Running map2alm on each slice')
    global map2alm_slices
    def map2alm_slices(i):
        result = hp.map2alm(binnedmaps[i],lmax=lmax,iter=1)
        result = result[idxpy2pixlmax]
        binnedmaps[i] = None
        return result
    pool = mp.Pool(processes=processes)
    binnedboostedalm = pool.map(map2alm_slices, range(nslices))
    pool.terminate()
    pool.join()
    map2alm_slices = None #Clear memory (preliminary results)

    binnedmaps = None #Clear memory (preliminary results)

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=binnedboostedalm[n].real*filtertablemerge[n]
            almslices_imag[n]=binnedboostedalm[n].imag*filtertablemerge[n]
            binnedboostedalm[n] = None

    almsmerged = almslices_real + almslices_imag*1j        
    almsmerged = np.sum(almsmerged,axis=0)
    almsmerged = reorder_idxpix2py(almsmerged)
    almslices_real = None; almslices_imag = None #Clear memory (preliminary results)

    end_time = time.time()
    print('Completed in: ',end_time-start_time,' seconds.')
    
    return almsmerged

def dipole_doppler_low_res_pol(alm_e,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=int(mp.cpu_count()/4),processes=int(4)):
    
    start_time = time.time()
    
    nslices = int(lmax/binsize)
    
    if 'dipole_doppler_map_pol' in globals():
        pass
    else:
        print('dipole doppler maps are not loaded yet, did you run the dipole_doppler_load function before?')
        return
    if almfile == True:
        almoriginal=hp.read_alm(alm_e)
    else:
        almoriginal = alm_e
    
    almoriginal=almoriginal[idxpy2pix]
    almoriginal=almoriginal[:hp.Alm.getsize(lmax)]

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=almoriginal.real*filtertable[n]
            almslices_imag[n]=almoriginal.imag*filtertable[n]
    almsliced = almslices_real + almslices_imag*1j
    almsliced = list(almsliced)

    almsize_tot = hp.Alm.getsize(lmax)
    almslicedpy_T = np.zeros(almsize_tot,dtype=np.complex128)
    almslicedpy_B = almslicedpy_T   
    mapInull = np.zeros(hp.nside2npix(nside))

    print('Running alm2map on each slice and applying dipole Doppler')
    global alm2map_slices
    def alm2map_slices(i):
        almslicedpy_E = almsliced[i][idxpix2py]
        resultI,resultQ,resultU = hp.alm2map((almslicedpy_T,almslicedpy_E,almslicedpy_B),nside=nside,lmax=lmax,verbose=False)
        resultQ = np.multiply(resultQ,dipole_doppler_map_pol[i])
        resultU = np.multiply(resultU,dipole_doppler_map_pol[i])
        almsliced[i] = None # Clear memory (preliminary results)
        return np.array([resultQ,resultU])
    pool = mp.Pool(processes=processes)
    binnedmaps = pool.map(alm2map_slices, range(nslices))
    pool.terminate()
    pool.join()
    alm2map_slices = None #Clear memory (preliminary results)

    binnedmaps = np.array(binnedmaps)
    print(binnedmaps[6,1].size) 
    print(binnedmaps.size)    

    print('Running map2alm on each slice')
    global map2alm_slices
    def map2alm_slices(i):
        resultT,resultE,resultB = hp.map2alm([mapInull,binnedmaps[i,0],binnedmaps[i,1]],lmax=lmax,iter=1)
        resultE = resultE[idxpy2pixlmax]
        binnedmaps[i] = None
        return resultE
    pool = mp.Pool(processes=processes)
    binnedboostedalm = pool.map(map2alm_slices, range(nslices))
    pool.terminate()
    pool.join()
    map2alm_slices = None #Clear memory (preliminary results)

    binnedmaps = None #Clear memory (preliminary results)

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=binnedboostedalm[n].real*filtertablemerge[n]
            almslices_imag[n]=binnedboostedalm[n].imag*filtertablemerge[n]
            binnedboostedalm[n] = None

    almsmerged = almslices_real + almslices_imag*1j        
    almsmerged = np.sum(almsmerged,axis=0)
    almsmerged = reorder_idxpix2py(almsmerged)
    almslices_real = None; almslices_imag = None #Clear memory (preliminary results)

    end_time = time.time()
    print('Completed in: ',end_time-start_time,' seconds.')
    
    return almsmerged

def dipole_doppler_low_res_pol_removal(alm_e,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=int(mp.cpu_count()/4),processes=int(4)):
    
    start_time = time.time()
    
    nslices = int(lmax/binsize)
    
    if 'dipole_doppler_map_pol' in globals():
        pass
    else:
        print('dipole doppler maps are not loaded yet, did you run the dipole_doppler_load function before?')
        return
    if almfile == True:
        almoriginal=hp.read_alm(alm_e)
    else:
        almoriginal = alm_e
    
    almoriginal=almoriginal[idxpy2pix]
    almoriginal=almoriginal[:hp.Alm.getsize(lmax)]

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=almoriginal.real*filtertable[n]
            almslices_imag[n]=almoriginal.imag*filtertable[n]
    almsliced = almslices_real + almslices_imag*1j
    almsliced = list(almsliced)

    almsize_tot = hp.Alm.getsize(lmax)
    almslicedpy_T = np.zeros(almsize_tot,dtype=np.complex128)
    almslicedpy_B = almslicedpy_T   
    mapInull = np.zeros(hp.nside2npix(nside))

    print('Running alm2map on each slice and applying dipole Doppler')
    global alm2map_slices
    def alm2map_slices(i):
        almslicedpy_E = almsliced[i][idxpix2py]
        resultI,resultQ,resultU = hp.alm2map((almslicedpy_T,almslicedpy_E,almslicedpy_B),nside=nside,lmax=lmax,verbose=False)
        resultQ = np.divide(resultQ,dipole_doppler_map_pol[i])
        resultU = np.divide(resultU,dipole_doppler_map_pol[i])
        almsliced[i] = None # Clear memory (preliminary results)
        return np.array([resultQ,resultU])
    pool = mp.Pool(processes=processes)
    binnedmaps = pool.map(alm2map_slices, range(nslices))
    pool.terminate()
    pool.join()
    alm2map_slices = None #Clear memory (preliminary results)

    binnedmaps = np.array(binnedmaps)
    print(binnedmaps[6,1].size) 
    print(binnedmaps.size)    

    print('Running map2alm on each slice')
    global map2alm_slices
    def map2alm_slices(i):
        resultT,resultE,resultB = hp.map2alm([mapInull,binnedmaps[i,0],binnedmaps[i,1]],lmax=lmax,iter=1)
        resultE = resultE[idxpy2pixlmax]
        binnedmaps[i] = None
        return resultE
    pool = mp.Pool(processes=processes)
    binnedboostedalm = pool.map(map2alm_slices, range(nslices))
    pool.terminate()
    pool.join()
    map2alm_slices = None #Clear memory (preliminary results)

    binnedmaps = None #Clear memory (preliminary results)

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=binnedboostedalm[n].real*filtertablemerge[n]
            almslices_imag[n]=binnedboostedalm[n].imag*filtertablemerge[n]
            binnedboostedalm[n] = None

    almsmerged = almslices_real + almslices_imag*1j        
    almsmerged = np.sum(almsmerged,axis=0)
    almsmerged = reorder_idxpix2py(almsmerged)
    almslices_real = None; almslices_imag = None #Clear memory (preliminary results)

    end_time = time.time()
    print('Completed in: ',end_time-start_time,' seconds.')
    
    return almsmerged

def dipole_doppler_low_res_weights(alm,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=int(mp.cpu_count()/4),processes=int(4)):
    
    start_time = time.time()
    
    nslices = int(lmax/binsize)
    
    if 'dipole_doppler_map' in globals():
        pass
    else:
        print('dipole doppler maps are not loaded yet, did you run the dipole_doppler_load function before?')
        return
    if almfile == True:
        almoriginal=hp.read_alm(alm)
    else:
        almoriginal = alm
    
    almoriginal=almoriginal[idxpy2pix]
    almoriginal=almoriginal[:hp.Alm.getsize(lmax)]

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=almoriginal.real*filtertable[n]
            almslices_imag[n]=almoriginal.imag*filtertable[n]
    almsliced = almslices_real + almslices_imag*1j
    almsliced = list(almsliced)
    
    print('Running alm2map on each slice and applying dipole Doppler')
    global alm2map_slices
    def alm2map_slices(i):
        almslicedpy=almsliced[i][idxpix2py]
        result = hp.alm2map(almslicedpy,nside=nside,lmax=lmax,verbose=False)
        result = np.multiply(result,dipole_doppler_map[i])
        almsliced[i] = None # Clear memory (preliminary results)
        return result
    pool = mp.Pool(processes=processes)
    binnedmaps = pool.map(alm2map_slices, range(nslices))
    pool.terminate()
    pool.join()
    alm2map_slices = None #Clear memory (preliminary results)

    print('Running map2alm on each slice')
    global map2alm_slices
    def map2alm_slices(i):
        result = hp.map2alm(binnedmaps[i],lmax=lmax,iter=0,use_weights=True)
        result = result[idxpy2pixlmax]
        binnedmaps[i] = None
        return result
    pool = mp.Pool(processes=processes)
    binnedboostedalm = pool.map(map2alm_slices, range(nslices))
    pool.terminate()
    pool.join()
    map2alm_slices = None #Clear memory (preliminary results)

    binnedmaps = None #Clear memory (preliminary results)

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=binnedboostedalm[n].real*filtertablemerge[n]
            almslices_imag[n]=binnedboostedalm[n].imag*filtertablemerge[n]
            binnedboostedalm[n] = None

    almsmerged = almslices_real + almslices_imag*1j        
    almsmerged = np.sum(almsmerged,axis=0)
    almsmerged = reorder_idxpix2py(almsmerged)
    almslices_real = None; almslices_imag = None #Clear memory (preliminary results)

    end_time = time.time()
    print('Completed in: ',end_time-start_time,' seconds.')
    
    return almsmerged

# dipole_doppler_low_res_removal remove the dipole doppler effect from a_lms (deboost)
# for parameters look at dipole_doppler_low_res function.
def dipole_doppler_low_res_removal(alm,almfile=False,lmax=2048,binsize=200,overbin=10,nside=2048,threads=int(mp.cpu_count()/4),processes=int(4)):
    
    start_time = time.time()

    nslices = int(lmax/binsize)
    
    if 'dipole_doppler_map' in globals():
        pass
    else:
        print('dipole doppler maps are not loaded yet, did you run the dipole_doppler_load function before?')
        return
    if almfile == True:
        almoriginal=hp.read_alm(alm)
    else:
        almoriginal = alm
    
    almoriginal=almoriginal[idxpy2pix]
    almoriginal=almoriginal[:hp.Alm.getsize(lmax)]

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=almoriginal.real*filtertable[n]
            almslices_imag[n]=almoriginal.imag*filtertable[n]
    almsliced = almslices_real + almslices_imag*1j
    almsliced = list(almsliced)
    
    print('Running alm2map on each slice and applying dipole Doppler')
    global alm2map_slices
    def alm2map_slices(i):
        almslicedpy=almsliced[i][idxpix2py]
        result = hp.alm2map(almslicedpy,nside=nside,lmax=lmax,verbose=False)
        result = np.divide(result,dipole_doppler_map[i])
        almsliced[i] = None #Clear memory (preliminary results)
        return result 
    pool = mp.Pool(processes=processes)
    binnedmaps = pool.map(alm2map_slices, range(nslices))
    pool.terminate()
    pool.join()
    alm2map_slices = None #Clear memory (preliminary results)

    print('Running map2alm on each slice')
    global map2alm_slices
    def map2alm_slices(i):
        result = hp.map2alm(binnedmaps[i],lmax=lmax,iter=1)
        result = result[idxpy2pixlmax]
        binnedmaps[i] = None #Clear memory (preliminary results)
        return result
    pool = mp.Pool(processes=processes)
    binnedboostedalm = pool.map(map2alm_slices, range(nslices))
    pool.terminate()
    pool.join()
    map2alm_slices = None #Clear memory (preliminary results)

    binnedmaps = None #Clear memory (preliminary results)

    almslices_real=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    almslices_imag=pymp.shared.array((nslices,hp.Alm.getsize(lmax)), dtype='float64')
    with pymp.Parallel(threads) as p:
        for n in p.range(nslices):
            almslices_real[n]=binnedboostedalm[n].real*filtertablemerge[n]
            almslices_imag[n]=binnedboostedalm[n].imag*filtertablemerge[n]
            binnedboostedalm[n] = None

    almsmerged = almslices_real + almslices_imag*1j        
    almsmerged = np.sum(almsmerged,axis=0)
    almsmerged = reorder_idxpix2py(almsmerged)
    almslices_real = None; almslices_imag = None

    end_time = time.time()
    print('Completed in: ',end_time-start_time,' seconds.')
    
    return almsmerged

# apply Doppler modulation on a alm
def apply_doppler(original_alm, beta_var, nside_var,lmax_var,latdir,longdir,threads_var,iter_var=1,verbose=True):
    if verbose == True: print('remember to set openmp threads limit equal to threads_var')
    doppler_boosted_map = hp.alm2map(original_alm,nside=nside_var,verbose=False)*doppler_boost_map_dir_s(beta_var, nside_var,latdir,longdir,threads_var)
    doppler_boosted_alm = hp.map2alm(doppler_boosted_map,lmax=lmax_var,iter=iter_var)
    return doppler_boosted_alm

# apply Aberration modulation on a alm (only on north direction) -> Needs fortran healpix boost mod version.]
# You need to set the "tmp_files_path" where the results of fortran will save tmp data.*
# This is very io bound if you use high l (>2000) -> Use the faster storage you have (nvme if possible)
def apply_aberration(original_alm,beta_var,nside_var,lmax_var,threads_var,iter_var=1,verbose=True,healpixboost_dir='',sufix='tmp',tmp_files_path='/home/pedro/hp_nvme/ab_boost_tmp_files/',save_tmp=False):
    hp.write_alm(str(tmp_files_path)+'original_alm_'+sufix+'.fits',original_alm,overwrite=True)
    file1 = open(str(tmp_files_path)+'params_'+sufix+'.dat','w') 
    file1.write('''
simul_type = 1
ab_dopp = 3
beta = '''+str(beta_var)+'''
nsmax = '''+str(nside_var)+'''
nlmax = '''+str(lmax_var)+'''
infile = ''
iseed = -1
fwhm_arcmin = 0
beam_file = ''
almsfile = '''+str(tmp_files_path)+'''original_alm_'''+sufix+'''.fits
plmfile = ''
outfile = !'''+str(tmp_files_path)+'''ab_boosted_map_'''+str(sufix)+'''.fits
outfile_alms = ''
''')
    file1.close() 
    subprocess.call(str(healpixboost_dir)+'synfastboost_v2 --double '+str(tmp_files_path)+'params_'+sufix+'.dat', shell=True)
    ab_map = hp.read_map(''+str(tmp_files_path)+'ab_boosted_map_'+str(sufix)+'.fits',verbose=False) 
    ab_alm = hp.map2alm(ab_map,lmax=lmax_var,iter=iter_var)
    if save_tmp == False:
    	subprocess.call('rm '+str(tmp_files_path)+'ab_boosted_map_'+str(sufix)+'.fits', shell=True)
    	subprocess.call('rm '+str(tmp_files_path)+'original_alm_'+sufix+'.fits', shell=True)
    return ab_alm

def rotate_alm_healpix(alm_path,lat_var,long_var,lmax,nside,sufix):
	lat_var=48.253 # Rotating to dipole direction
	long_var=264.021
	euler_lat=(90-lat_var)*(np.pi/180)
	euler_long=(long_var-360)*(np.pi/180)

	file1 = open('alteraalm_script'+str(sufix)+'.sh','w') 
	 
	file1.write('''
	#!/bin/bash

	echo "simul_type = 1
	infile = '''+str(alm_path)+'''
	nlmax = '''+str(lmax)+'''
	maskfile = ''
	theta_cut_deg =    0.0000000000000000
	regression = 0
	plmfile = ''
	outfile = ''
	outfile_alms = !almtest'''+str(sufix)+'''.fits
	iter_order = 3 " > anafast_params'''+str(sufix)+'''.sh

	wait
	anafastboost --double anafast_params'''+str(sufix)+'''.sh
	wait
		    
	echo "infile_alms = almtest'''+str(sufix)+'''.fits
	fwhm_arcmin_in =    0.00000000000000
	fwhm_arcmin_out =    0.00000000000000
	beam_file_out = ''
	epoch_in =    2000.0000000000000
	epoch_out =    2000.0000000000000
	DoWigner = T
	phi1 = 0        # For the Dipole direction, (dipole goes back to original position)
	theta ='''+str(euler_lat)+''' 
	phi2 ='''+str(euler_long)+''' 
	# phi1 = 1.6755   # For the Dipole direction, inverse rotation  (dipole becomes the new north)
	# theta = -0.7285  
	# phi2 = 0
	nsmax_out = '''+str(nside)+'''
	nlmax_out = '''+str(lmax)+'''
	outfile_alms = !rotated_alms'''+str(sufix)+'''.fits" > parametros_alteraalm'''+str(sufix)+'''.sh

	wait
	alteralmboost --double parametros_alteraalm'''+str(sufix)+'''.sh
	wait

	''')
	file1.close() 

	bashscript=subprocess.Popen("sh ./alteraalm_script"+str(sufix)+".sh", shell=True)
	bashscript.wait()

def expected_error_realistic(cl,clfile=False,fsky=1,nlfile=False,arcminbeam=0,lmin=3,lmax=2002,binsize=10,sufixname='test',threads=mp.cpu_count(),lmin_plot=1400,lmax_plot=2000,export_fg=False,export_sumbottom=False,export_plot=True):
   
    start = time.time()
    lmax=int(lmax+1)
    lmaxalm=int(lmax+1)
    ########## IMPORTING DATA ##########
    if clfile==True:
        cl=np.array((pyfits.open(cl))[1].data['TEMPERATURE'],dtype='float64')
    if nlfile != False:
        nl=hp.read_cl(nlfile)
        nl=nl[:lmax+2]
        cl2=cl[:lmax+2]
        cl = cl2+nl
    if nlfile == False:
        cl=cl[:lmax+2]
        cl2=cl  
    if arcminbeam != 0:   
        gaussianbeam=hp.gauss_beam(radians(arcminbeam/60),lmax=lmax)
        gaussianbeam2=gaussianbeam*gaussianbeam
        cl2=cl2*gaussianbeam2
    global cl_conv
    cl_conv = cl

    ########## CALCULATING c+ AND c- ##########
    cps0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmaxalm+1):
            idxl=l*(l+1)
            for m in range(-l,l+1):
                idxlm=idxl+m
                cps0=cps0factor( l, m)
                cms0=cms0factor( l, m)
                cps1=cps1factor( l, m)
                cms1=cms1factor( l, m)
                cps0abfactorlist[idxlm]=(l+2)*cps0
                cms0abfactorlist[idxlm]=(l-1)*cms0
                cps1abfactorlist[idxlm]=(l+2)*cps1
                cms1abfactorlist[idxlm]=(l-1)*cms1
                cps0doppfactorlist[idxlm]=-cps0
                cms0doppfactorlist[idxlm]=cms0
                cps1doppfactorlist[idxlm]=-cps1
                cms1doppfactorlist[idxlm]=cms1
    cms0abdoppfactorlist=cms0doppfactorlist+cms0abfactorlist
    cms1abdoppfactorlist=cms1doppfactorlist+cms1abfactorlist
    cps0abdoppfactorlist=cps0doppfactorlist+cps0abfactorlist
    cps1abdoppfactorlist=cps1doppfactorlist+cps1abfactorlist

    ########## CALCULATING <f> AND <g> (h) ##########
    symarraysize=hp.Alm.getsize(lmax=lmax)*2-lmax-2
    notsymarraysize=hp.Alm.getsize(lmax=lmax)
    
    global f0ab_theo, f0dopp_theo, f0abdopp_theo
    global f02ab_theo, f02dopp_theo, f02abdopp_theo
    global f1ab_theo, f1dopp_theo, f1abdopp_theo
    global g1ab_theo, g1dopp_theo, g1abdopp_theo
    f0ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f1ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')

    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1);idxlp1=(l+1)*(l+2);idxld2=int(idxl/2);lp1=l+1;
            for m in range(1,l):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl2[l]+cps0abfactorlist[idxlm]*cl2[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl2[l]+cps0doppfactorlist[idxlm]*cl2[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2[l]-cps1abfactorlist[idxlm]*cl2[lp1])
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2[l]-cps1doppfactorlist[idxlm]*cl2[lp1])
            for m in range(-l+1,0):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2[l]-cps1abfactorlist[idxlm]*cl2[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2[l]-cps1doppfactorlist[idxlm]*cl2[lp1])
            for m in [l]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl2[l]+cps0abfactorlist[idxlm]*cl2[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl2[l]+cps0doppfactorlist[idxlm]*cl2[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
            for m in [0]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f0ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl2[l]+cps0abfactorlist[idxlm]*cl2[lp1])
                f02ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl2[l]+cps0abfactorlist[idxlm]*cl2[lp1])
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2[l]-cps1abfactorlist[idxlm]*cl2[lp1])
                f0dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl2[l]+cps0doppfactorlist[idxlm]*cl2[lp1])
                f02dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl2[l]+cps0doppfactorlist[idxlm]*cl2[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2[l]-cps1doppfactorlist[idxlm]*cl2[lp1])

    f0abdopp_theo=f0ab_theo+f0dopp_theo
    f1abdopp_theo=f1ab_theo+f1dopp_theo
    f02abdopp_theo=f02ab_theo+f02dopp_theo

    sqrt2m=1/sqrt(2)
    f1ab_theo=sqrt2m*f1ab_theo
    f1dopp_theo=sqrt2m*f1dopp_theo
    f1abdopp_theo=sqrt2m*f1abdopp_theo
    g1ab_theo=f1ab_theo
    g1dopp_theo=f1dopp_theo
    g1abdopp_theo=f1abdopp_theo
    
    #Dealocating memory
    cps0abfactorlist=None;cms0abfactorlist=None;cps1abfactorlist=None;cms1abfactorlist=None;
    cps0doppfactorlist=None;cms0doppfactorlist=None;cps1doppfactorlist=None;cms1doppfactorlist=None;
    cms0abdoppfactorlist=None;cms1abdoppfactorlist=None;cps0abdoppfactorlist=None;cps1abdoppfactorlist=None;

    ########## CALCULATING Cl*Cl+1 ##########
    global clclplus1
    clclplus1=cl[1:][0:lmax+2]*cl[:-1][0:lmax+2]

    ########## CALCULATING bottom sum ##########
    global sumbottomzab, sumbottomxab, sumbottomyab
    global sumbottomzdopp, sumbottomxdopp, sumbottomydopp
    global sumbottomzabdopp, sumbottomxabdopp, sumbottomyabdopp
    sumbottomzab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomydopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmin,lmax):
            idxl=int(l*(l+1));idxld2=int(idxl/2);
            sumbottomzab[l-lmin]=np.sum((f02ab_theo[idxld2:idxld2+l+1]*f0ab_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxab[l-lmin]=np.sum((f1ab_theo[idxl-l:idxl+l+1]*f1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomyab[l-lmin]=np.sum((g1ab_theo[idxl-l:idxl+l+1]*g1ab_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomzdopp[l-lmin]=np.sum((f02dopp_theo[idxld2:idxld2+l+1]*f0dopp_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxdopp[l-lmin]=np.sum((f1dopp_theo[idxl-l:idxl+l+1]*f1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomydopp[l-lmin]=np.sum((g1dopp_theo[idxl-l:idxl+l+1]*g1dopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomzabdopp[l-lmin]=np.sum((f02abdopp_theo[idxld2:idxld2+l+1]*f0abdopp_theo[idxld2:idxld2+l+1]/clclplus1[l]))
            sumbottomxabdopp[l-lmin]=np.sum((f1abdopp_theo[idxl-l:idxl+l+1]*f1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))
            sumbottomyabdopp[l-lmin]=np.sum((g1abdopp_theo[idxl-l:idxl+l+1]*g1abdopp_theo[idxl-l:idxl+l+1]/clclplus1[l]))

    sumbottomab=np.array([sumbottomxab,sumbottomyab,sumbottomzab])
    sumbottomdopp=np.array([sumbottomxdopp,sumbottomydopp,sumbottomzdopp])
    sumbottomabdopp=np.array([sumbottomxabdopp,sumbottomyabdopp,sumbottomzabdopp])
    
    global inversevarianceabxbin, inversevarianceabybin, inversevarianceabzbin
    global inversevariancedoppxbin, inversevariancedoppybin, inversevariancedoppzbin
    global inversevarianceabdoppxbin, inversevarianceabdoppybin, inversevarianceabdoppzbin
    inversevarianceabxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')

    with pymp.Parallel(threads) as p:
        for i in p.range(int((lmax-lmin)/binsize)):
            inversevarianceabxbin[i]=(np.sum(sumbottomxab[i*binsize:(i+1)*binsize]))
            inversevarianceabybin[i]=(np.sum(sumbottomyab[i*binsize:(i+1)*binsize]))
            inversevarianceabzbin[i]=(np.sum(sumbottomzab[i*binsize:(i+1)*binsize]))
            inversevariancedoppxbin[i]=(np.sum(sumbottomxdopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppybin[i]=(np.sum(sumbottomydopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppzbin[i]=(np.sum(sumbottomzdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppxbin[i]=(np.sum(sumbottomxabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppybin[i]=(np.sum(sumbottomyabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppzbin[i]=(np.sum(sumbottomzabdopp[i*binsize:(i+1)*binsize]))

    if export_fg == True:
        np.savetxt(sufixname+'_'+'f02ab_theo.dat',f02ab_theo)
        np.savetxt(sufixname+'_'+'f1ab_theo.dat',f1ab_theo)
        np.savetxt(sufixname+'_'+'g1ab_theo.dat',g1ab_theo)
        np.savetxt(sufixname+'_'+'f02dopp_theo.dat',f02dopp_theo)
        np.savetxt(sufixname+'_'+'f1dopp_theo.dat',f1dopp_theo)
        np.savetxt(sufixname+'_'+'g1dopp_theo.dat',g1dopp_theo)
        np.savetxt(sufixname+'_'+'f02abdopp_theo.dat',f02abdopp_theo)
        np.savetxt(sufixname+'_'+'f1abdopp_theo.dat',f1abdopp_theo)
        np.savetxt(sufixname+'_'+'g1abdopp_theo.dat',g1abdopp_theo)

    if export_sumbottom == True:    
        np.savetxt(sufixname+'_'+'sumbottom_ab_sim_'+str(n+1)+'.dat',sumbottomab)
        np.savetxt(sufixname+'_'+'sumbottom_dopp_sim_'+str(n+1)+'.dat',sumbottomxdopp)
        np.savetxt(sufixname+'_'+'sumbottom_abdopp_sim_'+str(n+1)+'.dat',sumbottomabdopp)
    
    global stdabz_cumsum, stddoppz_cumsum, stdabdoppz_cumsum
    c=299792.458
    stdabz_cumsum=(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzab)))*c
    stddoppz_cumsum=(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzdopp)))*c
    stdabdoppz_cumsum=(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzabdopp)))*c
    plt.plot(np.arange(lmin_plot,lmax_plot),stdabz_cumsum[lmin_plot-1:lmax_plot-1], label='Ab')
    plt.plot(np.arange(lmin_plot,lmax_plot),stddoppz_cumsum[lmin_plot-1:lmax_plot-1], label='Dopp')
    plt.plot(np.arange(lmin_plot,lmax_plot),stdabdoppz_cumsum[lmin_plot-1:lmax_plot-1], label='AbDopp')
    plt.title('Theoretical Standard Deviation')
    plt.ylabel('\u03C3 (km/s)')
    plt.xlabel('\u2113')
    plt.legend()
    plt.grid()
    plt.show()
    if export_plot == True:
        plt.savefig(sufixname+'_plot.pdf')
    
    end = time.time()
    print('Completed in: ',end-start,' seconds.')

    return np.array([stdabz_cumsum,stddoppz_cumsum,stdabdoppz_cumsum]) , np.array([inversevarianceabzbin,inversevariancedoppzbin,inversevarianceabdoppzbin])

def expected_error_realistic_crossed(cl,clp,clpt,fsky=1,nlfile=False,nlfilep=False,arcminbeam=0,lmin=3,lmax=2002,binsize=10,sufixname='test',threads=mp.cpu_count(),lmin_plot=1400,lmax_plot=2000,export_fg=False,export_sumbottom=False,export_plot=True):
   
    start = time.time()
    lmax=int(lmax+1)
    lmaxalm=int(lmax+1)
    ########## IMPORTING DATA ##########
    if nlfile != False:
        nl=hp.read_cl(nlfile)
        nl=nl[:lmax+2]
        nlp=hp.read_cl(nlfilep)
        nlp=nlp[:lmax+2]
        cl2=cl[:lmax+2]
        cl2p=clp[:lmax+2]
        cl2pt=clpt[:lmax+2]
        cl = cl2+nl
        clp = cl2p+nlp
    if nlfile == False:
        cl=cl[:lmax+2]
        cl2=cl 
        clp=clp[:lmax+2]
        cl2p=clp  
        clpt=clpt[:lmax+2]
        cl2pt=clpt  
    global cl_conv
    cl_conv = cl

    ########## CALCULATING c+ AND c- ##########
    cps0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1abfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms0doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cps1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    cms1doppfactorlist=pymp.shared.array((hp.Alm.getsize(lmax=lmaxalm)*2-lmaxalm,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmaxalm+1):
            idxl=l*(l+1)
            for m in range(-l,l+1):
                idxlm=idxl+m
                cps0=cps0factor( l, m)
                cms0=cms0factor( l, m)
                cps1=cps1factor( l, m)
                cms1=cms1factor( l, m)
                cps0abfactorlist[idxlm]=(l+2)*cps0
                cms0abfactorlist[idxlm]=(l-1)*cms0
                cps1abfactorlist[idxlm]=(l+2)*cps1
                cms1abfactorlist[idxlm]=(l-1)*cms1
                cps0doppfactorlist[idxlm]=-cps0
                cms0doppfactorlist[idxlm]=cms0
                cps1doppfactorlist[idxlm]=-cps1
                cms1doppfactorlist[idxlm]=cms1
    cms0abdoppfactorlist=cms0doppfactorlist+cms0abfactorlist
    cms1abdoppfactorlist=cms1doppfactorlist+cms1abfactorlist
    cps0abdoppfactorlist=cps0doppfactorlist+cps0abfactorlist
    cps1abdoppfactorlist=cps1doppfactorlist+cps1abfactorlist

    ########## CALCULATING <f> AND <g> (h) ##########
    symarraysize=hp.Alm.getsize(lmax=lmax)*2-lmax-2
    notsymarraysize=hp.Alm.getsize(lmax=lmax)
    
    global f0ab_theo, f0dopp_theo, f0abdopp_theo
    global f02ab_theo, f02dopp_theo, f02abdopp_theo
    global f1ab_theo, f1dopp_theo, f1abdopp_theo
    global g1ab_theo, g1dopp_theo, g1abdopp_theo
    f0ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f0abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f02abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax),), dtype='float64')
    f1ab_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1dopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')
    f1abdopp_theo=pymp.shared.array((hp.Alm.getsize(lmax=lmax)*2-lmax,), dtype='float64')

    with pymp.Parallel(threads) as p:
        for l in p.range(lmax+1):
            idxl=l*(l+1);idxlp1=(l+1)*(l+2);idxld2=int(idxl/2);lp1=l+1;
            for m in range(1,l):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl2pt[l]+cps0abfactorlist[idxlm]*cl2pt[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl2pt[l]+cps0doppfactorlist[idxlm]*cl2pt[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2pt[l]-cps1abfactorlist[idxlm]*cl2pt[lp1])
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2pt[l]-cps1doppfactorlist[idxlm]*cl2pt[lp1])
            for m in range(-l+1,0):
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2pt[l]-cps1abfactorlist[idxlm]*cl2pt[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2pt[l]-cps1doppfactorlist[idxlm]*cl2pt[lp1])
            for m in [l]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                abcoef=(cms0abfactorlist[idxlp1m]*cl2pt[l]+cps0abfactorlist[idxlm]*cl2pt[lp1])
                doppcoef=(cms0doppfactorlist[idxlp1m]*cl2pt[l]+cps0doppfactorlist[idxlm]*cl2pt[lp1])
                f0ab_theo[idxld2+m]=abcoef
                f02ab_theo[idxld2+m]=2*abcoef
                f0dopp_theo[idxld2+m]=doppcoef
                f02dopp_theo[idxld2+m]=2*doppcoef
            for m in [0]:
                idxlm=idxl+m;idxlp1m=idxlp1+m;
                f0ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl2pt[l]+cps0abfactorlist[idxlm]*cl2pt[lp1])
                f02ab_theo[idxld2+m]=(cms0abfactorlist[idxlp1m]*cl2pt[l]+cps0abfactorlist[idxlm]*cl2pt[lp1])
                f1ab_theo[idxlm]=(cms1abfactorlist[idxlp1m+1]*cl2pt[l]-cps1abfactorlist[idxlm]*cl2pt[lp1])
                f0dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl2pt[l]+cps0doppfactorlist[idxlm]*cl2pt[lp1])
                f02dopp_theo[idxld2+m]=(cms0doppfactorlist[idxlp1m]*cl2pt[l]+cps0doppfactorlist[idxlm]*cl2pt[lp1])
                f1dopp_theo[idxlm]=(cms1doppfactorlist[idxlp1m+1]*cl2pt[l]-cps1doppfactorlist[idxlm]*cl2pt[lp1])

    f0abdopp_theo=f0ab_theo+f0dopp_theo
    f1abdopp_theo=f1ab_theo+f1dopp_theo
    f02abdopp_theo=f02ab_theo+f02dopp_theo

    sqrt2m=1/sqrt(2)
    f1ab_theo=sqrt2m*f1ab_theo
    f1dopp_theo=sqrt2m*f1dopp_theo
    f1abdopp_theo=sqrt2m*f1abdopp_theo
    g1ab_theo=f1ab_theo
    g1dopp_theo=f1dopp_theo
    g1abdopp_theo=f1abdopp_theo
    
    #Dealocating memory
    cps0abfactorlist=None;cms0abfactorlist=None;cps1abfactorlist=None;cms1abfactorlist=None;
    cps0doppfactorlist=None;cms0doppfactorlist=None;cps1doppfactorlist=None;cms1doppfactorlist=None;
    cms0abdoppfactorlist=None;cms1abdoppfactorlist=None;cps0abdoppfactorlist=None;cps1abdoppfactorlist=None;

    ########## CALCULATING bottom sum ##########
    global sumbottomzab, sumbottomxab, sumbottomyab
    global sumbottomzdopp, sumbottomxdopp, sumbottomydopp
    global sumbottomzabdopp, sumbottomxabdopp, sumbottomyabdopp
    sumbottomzab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyab=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomydopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomzabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomxabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    sumbottomyabdopp=pymp.shared.array((lmax-lmin,), dtype='float64')
    with pymp.Parallel(threads) as p:
        for l in p.range(lmin,lmax):
            idxl=int(l*(l+1));idxld2=int(idxl/2);
            sumbottomzab[l-lmin]=np.sum((f02ab_theo[idxld2:idxld2+l+1]*f0ab_theo[idxld2:idxld2+l+1]/(cl[l]*clp[l+1])))
            sumbottomxab[l-lmin]=np.sum((f1ab_theo[idxl-l:idxl+l+1]*f1ab_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))
            sumbottomyab[l-lmin]=np.sum((g1ab_theo[idxl-l:idxl+l+1]*g1ab_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))
            sumbottomzdopp[l-lmin]=np.sum((f02dopp_theo[idxld2:idxld2+l+1]*f0dopp_theo[idxld2:idxld2+l+1]/(cl[l]*clp[l+1])))
            sumbottomxdopp[l-lmin]=np.sum((f1dopp_theo[idxl-l:idxl+l+1]*f1dopp_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))
            sumbottomydopp[l-lmin]=np.sum((g1dopp_theo[idxl-l:idxl+l+1]*g1dopp_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))
            sumbottomzabdopp[l-lmin]=np.sum((f02abdopp_theo[idxld2:idxld2+l+1]*f0abdopp_theo[idxld2:idxld2+l+1]/(cl[l]*clp[l+1])))
            sumbottomxabdopp[l-lmin]=np.sum((f1abdopp_theo[idxl-l:idxl+l+1]*f1abdopp_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))
            sumbottomyabdopp[l-lmin]=np.sum((g1abdopp_theo[idxl-l:idxl+l+1]*g1abdopp_theo[idxl-l:idxl+l+1]/(cl[l]*clp[l+1])))

    sumbottomab=np.array([sumbottomxab,sumbottomyab,sumbottomzab])
    sumbottomdopp=np.array([sumbottomxdopp,sumbottomydopp,sumbottomzdopp])
    sumbottomabdopp=np.array([sumbottomxabdopp,sumbottomyabdopp,sumbottomzabdopp])
    
    global inversevarianceabxbin, inversevarianceabybin, inversevarianceabzbin
    global inversevariancedoppxbin, inversevariancedoppybin, inversevariancedoppzbin
    global inversevarianceabdoppxbin, inversevarianceabdoppybin, inversevarianceabdoppzbin
    inversevarianceabxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevariancedoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppxbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppybin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')
    inversevarianceabdoppzbin=pymp.shared.array((int((lmax-lmin)/binsize),), dtype='float64')

    with pymp.Parallel(threads) as p:
        for i in p.range(int((lmax-lmin)/binsize)):
            inversevarianceabxbin[i]=(np.sum(sumbottomxab[i*binsize:(i+1)*binsize]))
            inversevarianceabybin[i]=(np.sum(sumbottomyab[i*binsize:(i+1)*binsize]))
            inversevarianceabzbin[i]=(np.sum(sumbottomzab[i*binsize:(i+1)*binsize]))
            inversevariancedoppxbin[i]=(np.sum(sumbottomxdopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppybin[i]=(np.sum(sumbottomydopp[i*binsize:(i+1)*binsize]))
            inversevariancedoppzbin[i]=(np.sum(sumbottomzdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppxbin[i]=(np.sum(sumbottomxabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppybin[i]=(np.sum(sumbottomyabdopp[i*binsize:(i+1)*binsize]))
            inversevarianceabdoppzbin[i]=(np.sum(sumbottomzabdopp[i*binsize:(i+1)*binsize]))

    if export_fg == True:
        np.savetxt(sufixname+'_'+'f02ab_theo.dat',f02ab_theo)
        np.savetxt(sufixname+'_'+'f1ab_theo.dat',f1ab_theo)
        np.savetxt(sufixname+'_'+'g1ab_theo.dat',g1ab_theo)
        np.savetxt(sufixname+'_'+'f02dopp_theo.dat',f02dopp_theo)
        np.savetxt(sufixname+'_'+'f1dopp_theo.dat',f1dopp_theo)
        np.savetxt(sufixname+'_'+'g1dopp_theo.dat',g1dopp_theo)
        np.savetxt(sufixname+'_'+'f02abdopp_theo.dat',f02abdopp_theo)
        np.savetxt(sufixname+'_'+'f1abdopp_theo.dat',f1abdopp_theo)
        np.savetxt(sufixname+'_'+'g1abdopp_theo.dat',g1abdopp_theo)

    if export_sumbottom == True:    
        np.savetxt(sufixname+'_'+'sumbottom_ab_sim_'+str(n+1)+'.dat',sumbottomab)
        np.savetxt(sufixname+'_'+'sumbottom_dopp_sim_'+str(n+1)+'.dat',sumbottomxdopp)
        np.savetxt(sufixname+'_'+'sumbottom_abdopp_sim_'+str(n+1)+'.dat',sumbottomabdopp)
    
    global stdabz_cumsum, stddoppz_cumsum, stdabdoppz_cumsum
    c=299792.458
    stdabz_cumsum=(1/np.sqrt(2))*(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzab)))*c
    stddoppz_cumsum=(1/np.sqrt(2))*(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzdopp)))*c
    stdabdoppz_cumsum=(1/np.sqrt(2))*(1/np.sqrt(fsky))*np.sqrt(1/(np.cumsum(sumbottomzabdopp)))*c
    plt.plot(np.arange(lmin_plot,lmax_plot),stdabz_cumsum[lmin_plot-1:lmax_plot-1], label='Ab')
    plt.plot(np.arange(lmin_plot,lmax_plot),stddoppz_cumsum[lmin_plot-1:lmax_plot-1], label='Dopp')
    plt.plot(np.arange(lmin_plot,lmax_plot),stdabdoppz_cumsum[lmin_plot-1:lmax_plot-1], label='AbDopp')
    plt.title('Theoretical Standard Deviation')
    plt.ylabel('\u03C3 (km/s)')
    plt.xlabel('\u2113')
    plt.legend()
    plt.grid()
    plt.show()
    if export_plot == True:
        plt.savefig(sufixname+'_plot.pdf')
    
    end = time.time()
    print('Completed in: ',end-start,' seconds.')

    return np.array([stdabz_cumsum,stddoppz_cumsum,stdabdoppz_cumsum]) , np.array([inversevarianceabzbin,inversevariancedoppzbin,inversevarianceabdoppzbin])

def masksymm(mask,nside_out,fwhm_arcmin=10,gibbs_removal=True,iter=3):
    print('This function only works if the inputed mask is not apodized. The inputed mask have to be only 0s and 1s.')

    #Reading mask
    mask = hp.ud_grade(mask,nside_out=nside_out) # Changing nside, if needed
    
    #Getting mask specs
    npix=len(mask)
    nside=hp.npix2nside(npix)
    
    #Inverting mask
    mask=mask+1
    mask[mask!=1]=0
    
    #Getting normal (pixcoord) and antipodal(pixcoordapo) coordinates
    pixidx = np.arange(int(npix/2))
    long_lat_array = np.array(hp.pix2ang(nside,pixidx,lonlat=True))
    pixcoordapo = ((long_lat_array+np.array([np.repeat(180,int(npix/2)),np.zeros(int(npix/2))]))*np.array([np.ones(int(npix/2)),-1*np.ones(int(npix/2))])).T
    npixapo=hp.ang2pix(nside,pixcoordapo[:,0],pixcoordapo[:,1],lonlat=True)
    
    #Creating symmetric mask
    masksymm = mask
    masksymm[npixapo]=mask[npixapo]+mask[pixidx]
    masksymm[pixidx]=mask[npixapo]
    masksymm[masksymm>1]=1
    masksymm=masksymm+1
    masksymm[masksymm>1]=0
    
    #Apodizing
    if fwhm_arcmin != 0:
        masksymm = hp.smoothing(masksymm,fwhm=np.deg2rad(fwhm_arcmin/60), iter=iter,verbose=False);
    if gibbs_removal==True:
        #Removing Gibbs Effect
        masksymm[masksymm>1]=1
        masksymm[masksymm<0]=0
    
    return masksymm

def mapI_wigner_rotation(map_orig,lat_var,long_var,lmax_var,rot_type='north2dir'):
    if rot_type == 'north2dir':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=True)
        map_rot = r.rotate_map_alms(map_orig,lmax=lmax_var)
    if rot_type == 'dir2north':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=False)
        map_rot = r.rotate_map_alms(map_orig,lmax=lmax_var)
    return map_rot
    
def almT_wigner_rotation(alm_orig,lat_var,long_var,rot_type='north2dir'):
    if rot_type == 'north2dir':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=True)
        alm_rot = r.rotate_alm(alm_orig)
    if rot_type == 'dir2north':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=False)
        alm_rot = r.rotate_alm(alm_orig)
    return alm_rot    

def mapQU_wigner_rotation(map_origI,map_origQ,map_origU,lat_var,long_var,lmax_var,rot_type='north2dir'):
    if rot_type == 'north2dir':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=True)
        map_rot = r.rotate_map_alms((map_origI,map_origQ,map_origU),lmax=lmax_var)
    if rot_type == 'dir2north':
        r = hp.Rotator(rot=[long_var-360,lat_var-90,0],inv=False)
        map_rot = r.rotate_map_alms((map_origI,map_origQ,map_origU),lmax=lmax_var)
    return map_rot

def abang(beta,colat):
    colat = colat -180
    return (180 - np.arccos((np.cos(colat*(np.pi/180))+beta)/(1+beta*np.cos(colat*(np.pi/180))))*(180/np.pi))*(np.pi/180)

def kernel_pixel_ab(nside_var,beta_var, threads_var=mp.cpu_count()):
    npix = hp.nside2npix(nside_var)
    pixarray = np.arange(npix)
    pixangles = (np.array([hp.pix2ang(nside=nside_var,ipix=pixarray)])[0]).T
    abkernel = pymp.shared.array((npix,), dtype='float64')    # Applying to a map.
    with pymp.Parallel(threads_var) as p:
        for pix in p.range(npix):
            pixcolat = pixangles[pix,0]*180/np.pi
            abkernel[pix] = abang(beta_var,pixcolat)
    return abkernel

def ab_boost_map_cached(orig_map,kernelab_cached, threads_var=mp.cpu_count()):
    nside_var = hp.npix2nside(orig_map.size)
    npix = hp.nside2npix(nside_var)
    pixarray = np.arange(npix)
    pixangles = (np.array([hp.pix2ang(nside=nside_var,ipix=pixarray)])[0]).T
    boostedmap = pymp.shared.array((npix,), dtype='float64')    # Applying to a map.
    with pymp.Parallel(threads_var) as p:
        for pix in p.range(npix):
            pixlong = pixangles[pix,1]
            boostedmap[pix] = hp.pixelfunc.get_interp_val(orig_map,kernelab_cached[pix],pixlong)
    return boostedmap

def kernel_pixel_ab_s(nside_var,beta_var, threads_var=mp.cpu_count()):
    npix = hp.nside2npix(nside_var)
    pixarray = np.arange(npix)
    pixangles = (np.array([hp.pix2ang(nside=nside_var,ipix=pixarray)])[0]).T
    abkernel = pymp.shared.array((npix,2), dtype='float64')    # Applying to a map.
    with pymp.Parallel(threads_var) as p:
        for pix in p.range(npix):
            pixcolat = pixangles[pix,0]*180/np.pi
            abkernel[pix,0] = abang(beta_var,pixcolat)
            abkernel[pix,1] = pixangles[pix,1]
    return abkernel

def ab_boost_map_cached_s(orig_map,kernelab_cached, threads_var=mp.cpu_count()):
    nside_var = hp.npix2nside(orig_map.size)
    npix = hp.nside2npix(nside_var)
    pixarray = np.arange(npix)
    boostedmap = hp.pixelfunc.get_interp_val(orig_map,kernelab_cached[:,0],kernelab_cached[:,1])
    return boostedmap

