# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import healpy as hp
import numpy as np
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

#%%
#cmap_test = colors.LinearSegmentedColormap.from_list("", ["red","orange","green","blue"])
#cmap_func = matplotlib.cm.get_cmap(cmap_test)
lmin = 200
c=299792.458

vec_l = np.array([1000,1200,1400,1600,1800])
vec_l_EE = np.array([1000,1150,1150,1150,1150])

cmap_test = colors.LinearSegmentedColormap.from_list("t", ["red","orange","green","blue","purple"])
cmap_func = matplotlib.cm.get_cmap(cmap_test)

abdopp_TT_vec_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l])

abdopp_TT_vec_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l])

abdopp_EE_vec_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

abdopp_EE_vec_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

#%%

x_stat_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
y_stat_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
z_stat_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

x_sist_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
y_sist_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
z_sist_error_EE_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

#%%
x_stat_error_EE_nilc = np.sqrt(x_stat_error_EE_nilc**2 + x_sist_error_EE_nilc**2)
y_stat_error_EE_nilc = np.sqrt(y_stat_error_EE_nilc**2 + y_sist_error_EE_nilc**2)
z_stat_error_EE_nilc = np.sqrt(z_stat_error_EE_nilc**2 + z_sist_error_EE_nilc**2)

#%%
x_stat_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
y_stat_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
z_stat_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_staterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

x_sist_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
y_sist_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])
z_sist_error_EE_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_EE_v2_EE_v2_w_DD_results_ab_dopp_abdopp_sim_estimator_final_sisterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l_EE])

#%%
x_stat_error_EE_smica = np.sqrt(x_stat_error_EE_smica**2 + x_sist_error_EE_smica**2)
y_stat_error_EE_smica = np.sqrt(y_stat_error_EE_smica**2 + y_sist_error_EE_smica**2)
z_stat_error_EE_smica = np.sqrt(z_stat_error_EE_smica**2 + z_sist_error_EE_smica**2)

#%%
x_stat_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
y_stat_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
z_stat_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])

x_sist_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
y_sist_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
z_sist_error_TT_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
#%%
x_stat_error_TT_nilc = np.sqrt(x_stat_error_TT_nilc**2 + x_sist_error_TT_nilc**2)
y_stat_error_TT_nilc = np.sqrt(y_stat_error_TT_nilc**2 + y_sist_error_TT_nilc**2)
z_stat_error_TT_nilc = np.sqrt(z_stat_error_TT_nilc**2 + z_sist_error_TT_nilc**2)

#%%
x_stat_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
y_stat_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
z_stat_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_staterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])

x_sist_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_x_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
y_sist_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_y_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])
z_sist_error_TT_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_sisterror_absunbiased_z_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2]/c for i in vec_l])

#%%
x_stat_error_TT_smica = np.sqrt(x_stat_error_TT_smica**2 + x_sist_error_TT_smica**2)
y_stat_error_TT_smica = np.sqrt(y_stat_error_TT_smica**2 + y_sist_error_TT_smica**2)
z_stat_error_TT_smica = np.sqrt(z_stat_error_TT_smica**2 + z_sist_error_TT_smica**2)

#%%

smica_abdopp_TTEE_x = np.array([[(abdopp_TT_vec_smica[i,0]/(x_stat_error_TT_smica[i]**2)+abdopp_EE_vec_smica[i,0]/(x_stat_error_EE_smica[i]**2))/(1/(x_stat_error_TT_smica[i]**2)+1/(x_stat_error_EE_smica[i]**2)) for i in range(vec_l.size)]])
smica_abdopp_TTEE_y = np.array([[(abdopp_TT_vec_smica[i,1]/(y_stat_error_TT_smica[i]**2)+abdopp_EE_vec_smica[i,1]/(y_stat_error_EE_smica[i]**2))/(1/(y_stat_error_TT_smica[i]**2)+1/(y_stat_error_EE_smica[i]**2)) for i in range(vec_l.size)]])
smica_abdopp_TTEE_z = np.array([[(abdopp_TT_vec_smica[i,2]/(z_stat_error_TT_smica[i]**2)+abdopp_EE_vec_smica[i,2]/(z_stat_error_EE_smica[i]**2))/(1/(z_stat_error_TT_smica[i]**2)+1/(z_stat_error_EE_smica[i]**2)) for i in range(vec_l.size)]])

nilc_abdopp_TTEE_x = np.array([[(abdopp_TT_vec_nilc[i,0]/(x_stat_error_TT_nilc[i]**2)+abdopp_EE_vec_nilc[i,0]/(x_stat_error_EE_nilc[i]**2))/(1/(x_stat_error_TT_nilc[i]**2)+1/(x_stat_error_EE_nilc[i]**2)) for i in range(vec_l.size)]])
nilc_abdopp_TTEE_y = np.array([[(abdopp_TT_vec_nilc[i,1]/(y_stat_error_TT_nilc[i]**2)+abdopp_EE_vec_nilc[i,1]/(y_stat_error_EE_nilc[i]**2))/(1/(y_stat_error_TT_nilc[i]**2)+1/(y_stat_error_EE_nilc[i]**2)) for i in range(vec_l.size)]])
nilc_abdopp_TTEE_z = np.array([[(abdopp_TT_vec_nilc[i,2]/(z_stat_error_TT_nilc[i]**2)+abdopp_EE_vec_nilc[i,2]/(z_stat_error_EE_nilc[i]**2))/(1/(z_stat_error_TT_nilc[i]**2)+1/(z_stat_error_EE_nilc[i]**2)) for i in range(vec_l.size)]])

abdopp_vec_smica = np.array([smica_abdopp_TTEE_x[0],smica_abdopp_TTEE_y[0],smica_abdopp_TTEE_z[0]])
abdopp_vec_nilc = np.array([nilc_abdopp_TTEE_x[0],nilc_abdopp_TTEE_y[0],nilc_abdopp_TTEE_z[0]])

abdopp_vec_smica = abdopp_vec_smica.T
abdopp_vec_nilc = abdopp_vec_nilc.T

np.linalg.norm(abdopp_vec_nilc,axis=1)*c
np.linalg.norm(abdopp_vec_smica,axis=1)*c

dipole_lat = 48.253
dipole_long = 264.021
dipole_beta = 0.00123357
fiducialvec = hp.ang2vec(dipole_long,dipole_lat,lonlat=True)*dipole_beta  

np.linalg.norm(abdopp_vec_nilc,axis=1)*c
np.linalg.norm(abdopp_vec_smica,axis=1)*c

np.dot(abdopp_vec_nilc,hp.ang2vec(dipole_long,dipole_lat,lonlat=True))*c
np.dot(abdopp_vec_smica,hp.ang2vec(dipole_long,dipole_lat,lonlat=True))*c
#%%
#vec_l = np.array([1000,1200,1400,1600,1800,2000])

nside=1024
m = np.zeros(hp.nside2npix(nside))

#coord = hp.query_disc(nside=nside,vec=np.array([1,1,-1]),radius=np.deg2rad(1.6),inclusive=True,fact=16)
#m[coord] = 2

def disk_plot(value,dir_vec,ang_size,nside_var):
    coord = hp.query_disc(nside=nside_var,vec=dir_vec,radius=np.deg2rad(ang_size),inclusive=True,fact=16)
    m[coord] = value   
def mollweid_vec(l,lmin,lmax,dir_vec,size_var,symbol,fill_style):
    lon,lat = hp.vec2ang(dir_vec,lonlat=True)
    hp.visufunc.projplot(lon,lat,symbol,lonlat=True,color=cmap_func((l-lmin)/(lmax-lmin)),markersize=size_var,fillstyle=fill_style)



#for i in range(5):
#    disk_plot(vec_l[i],ab_vec_nilc[i],0.1,nside)
    
#for i in range(5):
#    disk_plot(vec_l[i],dopp_vec_nilc[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],abdopp_vec_nilc[i],0.1,nside)
    
#for i in range(5):
#    disk_plot(vec_l[i],ab_vec_smica[i],0.1,nside)
    
#for i in range(5):
#    disk_plot(vec_l[i],dopp_vec_smica[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],abdopp_vec_smica[i],0.1,nside)
    
m[m==0] = hp.UNSEEN
hp.mask_bad(m)
hp.mollview(m,badcolor='white',unit=r'$\ell_{max}$',cbar=None,cmap=cmap_test,xsize=2048,title="")
hp.graticule()

########################################
########################################
phi = np.linspace(0, 2.*np.pi, 180)
#
#aberror=np.array([laterror_smica[0,0],longerror_smica[0,0]]);
#dopperror=np.array([laterror_smica[1,1],longerror_smica[1,1]]);

laterror_smica = np.loadtxt('data/smicaTTEE1sigmaelipsevalues.dat')[:,1]
longerror_smica = np.loadtxt('data/smicaTTEE1sigmaelipsevalues.dat')[:,0]

laterror_smica2 = np.loadtxt('data/smicaTTEE2sigmaelipsevalues.dat')[:,1]
longerror_smica2 = np.loadtxt('data/smicaTTEE2sigmaelipsevalues.dat')[:,0]

abdopperror=np.array([laterror_smica[2],longerror_smica[2]]);
abdopperror2=np.array([laterror_smica2[2],longerror_smica2[2]]);
#
#abvec = ab_vec_smica[4]
#ablon,ablat=hp.vec2ang(abvec,lonlat=True)
#doppvec = dopp_vec_smica[4]
#dopplon,dopplat=hp.vec2ang(-doppvec,lonlat=True)
abdoppvec = abdopp_vec_smica[vec_l.size-1]
abdopplon,abdopplat=hp.vec2ang(abdoppvec,lonlat=True)
marker='v';colorm='b';#title='S';
#ablongerror1=aberror[1];ablaterror1=aberror[0];colore1='grey';
#ablongerror2=2*aberror[1];ablaterror2=2*aberror[0];colore2='lightgrey'
#dopplongerror1=dopperror[1];dopplaterror1=dopperror[0];colore1='grey';
#dopplongerror2=2*dopperror[1];dopplaterror2=2*dopperror[0];colore2='lightgrey'
abdopplongerror1=abdopperror[1];abdopplaterror1=abdopperror[0];colore1='grey';
abdopplongerror2=abdopperror2[1];abdopplaterror2=abdopperror2[0];colore2='lightgrey'
#
#m.plot(x, y,marker,color=colorm,markersize=4,label='Aberration')
#r1 =longerror1; r2 = laterror1;
#x,y = m(lon + r1np.cos(phi) , lat + r2np.sin(phi))
#m.plot(x, y, color=colore1)
#r3 = longerror2; r4 = laterror2
#x,y = m(lon + r1np.cos(phi) , lat + r2np.sin(phi))
#m.plot(x, y, color=colore2)
def error_bar(lat_start,long_start,width,headwidth,headlength,r2,r1,r22s,r12s,ticks=False):
#   Setting the projection
#    proj = hp.projector.MollweideProj()
#    hp.visufunc.projplot(lon,lat,'^',lonlat=True,color='y', markersize=4)
    if ticks==True:
        hp.visufunc.projplot(long_start+r1*np.cos(phi),lat_start + (r2)*np.sin(phi),lonlat=True,color='purple',linestyle='dashed',linewidth=2)
        hp.visufunc.projplot(long_start+r12s*np.cos(phi),lat_start + (r22s)*np.sin(phi),lonlat=True,color='plum',linestyle='dashed',linewidth=2)    
    else:
        hp.visufunc.projplot(long_start+r1*np.cos(phi),lat_start + (r2)*np.sin(phi),lonlat=True,color='purple')
        hp.visufunc.projplot(long_start+r12s*np.cos(phi),lat_start + (r22s)*np.sin(phi),lonlat=True,color='plum')
#   Plotting vectors form (lat_start,long_start) coordinates to (lat_end,long_end) coordinates
#    plt.annotate('', xy=(proj.ang2xy(lat_end, long_end,lonlat=True)), xytext=(proj.ang2xy(lat_start, long_start,lonlat=True)),   arrowprops=dict(color=color,width=width,headwidth=headwidth,headlength=headlength))


error_bar(abdopplat,abdopplon,0.5,6,6,abdopplaterror1,abdopplongerror1,abdopplaterror2,abdopplongerror2)

#aberror=np.array([laterror_smica[0,0],longerror_smica[0,0]]);
#dopperror=np.array([laterror_smica[1,1],longerror_smica[1,1]]);

laterror_nilc = np.loadtxt('data/nilcTTEE1sigmaelipsevalues.dat')[:,1]
longerror_nilc = np.loadtxt('data/nilcTTEE1sigmaelipsevalues.dat')[:,0]

laterror_nilc2 = np.loadtxt('data/nilcTTEE2sigmaelipsevalues.dat')[:,1]
longerror_nilc2 = np.loadtxt('data/nilcTTEE2sigmaelipsevalues.dat')[:,0]

abdopperror=np.array([laterror_nilc[2],longerror_nilc[2]]);
abdopperror2=np.array([laterror_nilc2[2],longerror_nilc2[2]]);
#
#abvec = ab_vec_nilc[4]
#ablon,ablat=hp.vec2ang(abvec,lonlat=True)
#doppvec = dopp_vec_nilc[4]
#dopplon,dopplat=hp.vec2ang(-doppvec,lonlat=True)
abdoppvec = abdopp_vec_nilc[vec_l.size-1]
abdopplon,abdopplat=hp.vec2ang(abdoppvec,lonlat=True)
marker='v';colorm='b';#title='S';
#ablongerror1=aberror[1];ablaterror1=aberror[0];colore1='grey';
#ablongerror2=2*aberror[1];ablaterror2=2*aberror[0];colore2='lightgrey'
#dopplongerror1=dopperror[1];dopplaterror1=dopperror[0];colore1='grey';
#dopplongerror2=2*dopperror[1];dopplaterror2=2*dopperror[0];colore2='lightgrey'
abdopplongerror1=abdopperror[1];abdopplaterror1=abdopperror[0];colore1='grey';
abdopplongerror2=abdopperror2[1];abdopplaterror2=abdopperror2[0];colore2='lightgrey'
#
#error_bar(ablat,ablon,0.5,6,6,ablaterror1,ablongerror1)

#error_bar(dopplat,dopplon,0.5,6,6,dopplaterror1,dopplongerror1)

error_bar(abdopplat,abdopplon,0.5,6,6,abdopplaterror1,abdopplongerror1,abdopplaterror2,abdopplongerror2,ticks=True)


########################################
########################################

#for i in range(5):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],ab_vec_nilc[i],6,'s',fill_style='left')
    
#for i in range(5):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],-dopp_vec_nilc[i],6,'o',fill_style='left')
    
for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],abdopp_vec_nilc[i],11.5,'D',fill_style='left')
    
#for i in range(5):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],ab_vec_smica[i],6,'s',fill_style='right')
    
#for i in range(5):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],-dopp_vec_smica[i],6,'o',fill_style='right')

for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],abdopp_vec_smica[i],11.5,'D',fill_style='right')

dipole_lat = 48.253
dipole_long = 264.021
dipole_beta = 0.00123357
fiducialvec = hp.ang2vec(dipole_long,dipole_lat,lonlat=True)*dipole_beta   
hp.visufunc.projplot(dipole_long,dipole_lat,'*',lonlat=True,color='k',markersize=12.5)
    
plt.rcParams["legend.fontsize"] = 15.3
#legend1 = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
#                          markersize=10, label='',fillstyle='right')
#legend2 = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
#                          markersize=10, label='',fillstyle='right')
legend1 = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
                          markersize=15, label='',fillstyle='right')
#legend3 = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
#                          markersize=10, label='Aberration',fillstyle='left')
#legend4 = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
#                          markersize=10, label='Doppler',fillstyle='left')
legend2 = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
                          markersize=15, label='Pec. Velocity',fillstyle='left')
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
norm = matplotlib.colors.BoundaryNorm(vec_l, cmap_test.N)
cmap = fig.colorbar(image,orientation='horizontal',fraction=0.025,pad=0.01,aspect=50, ticks=vec_l)#,spacing='proportional', ticks=vec_l, boundaries=vec_l, format='%1i')
cmap.ax.tick_params(labelsize=18)

ax.text(-0.09, -1.38,r'$\ell_{max}$', fontsize=22)
#ax.text(-2.0, 0.78, 'TT+EE', fontsize=30)
ax.text(-2.0, 0.78, 'T', fontsize=30)
ax.text(-1.856, 0.78, 'T+EE', fontsize=30)

#ax.text(0.35, 0.37,'Doppler', fontsize=10.5,color='grey')
#ax.text(0.35, 0.135,'Doppler', fontsize=10.5,color='lightgrey')

#ax.text(1.2, 0.44,'Aberration', fontsize=10.5,color='grey')
#ax.text(1.3, 0.31,'Aberration', fontsize=10.5,color='lightgrey')

plt.legend(handles=[legend1, legend2],loc="lower right",title="SMICA       NILC", title_fontsize=15.3, ncol=2,handletextpad=1.24)._legend_box.align='left'

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

def coord_plot(long_start,lat_start,color_var,size_var,text):
    hp.visufunc.projtext(long_start,lat_start,text,lonlat=True,color=color_var,size=2*size_var)

coord_color='grey'
coord_color2='grey'
#coord_size=4
#coord_plot(0+5,1.5,coord_color,coord_size,'0')
#coord_plot(30+8,1.5,coord_color,coord_size,'30')
#coord_plot(60+8,1.5,coord_color,coord_size,'60')
#coord_plot(90+8,1.5,coord_color,coord_size,'90')
#coord_plot(120+11,1.5,coord_color,coord_size,'120')
#coord_plot(150+11,1.5,coord_color,coord_size,'150')
#coord_plot(180+11,1.5,coord_color,coord_size,'180')
#coord_plot(210+11,1.5,coord_color,coord_size,'210')
#coord_plot(240+11,1.5,coord_color,coord_size,'240')
#coord_plot(270+11,1.5,coord_color,coord_size,'270')
#coord_plot(300+11,1.5,coord_color,coord_size,'300')
#coord_plot(330+11,1.5,coord_color,coord_size,'330')
#coord_plot(180+5,0+1.5,coord_color,coord_size,'     0')
#coord_plot(180+5,30+1.5,coord_color,coord_size,'     30')
#coord_plot(180+5,60+1.5,coord_color,coord_size,'     60')
#coord_plot(180+5,-30-3,coord_color,coord_size,'      -30')
#coord_plot(180+5,-60-3,coord_color,coord_size,'         -60')
coord_size=8
coord_size2=6.5
coord_plot(0+5.9,1,coord_color2,coord_size2,'0')
coord_plot(30+11,1,coord_color2,coord_size2,'30')
coord_plot(60+11,1,coord_color2,coord_size2,'60')
coord_plot(90+11,1,coord_color2,coord_size2,'90')
coord_plot(120+15.9,1,coord_color2,coord_size2,'120')
coord_plot(150+15.9,1,coord_color2,coord_size2,'150')
coord_plot(180+16,1,coord_color2,coord_size2,'180')
coord_plot(210+16,1,coord_color2,coord_size2,'210')
coord_plot(240+16,1,coord_color2,coord_size2,'240')
coord_plot(270+16,1,coord_color2,coord_size2,'270')
coord_plot(300+16,1,coord_color2,coord_size2,'300')
coord_plot(330+16,1,coord_color2,coord_size2,'330')
coord_plot(180,0-3,coord_color,coord_size,' 0')
coord_plot(180,30-2,coord_color,coord_size,' 30')
coord_plot(180,60-2,coord_color,coord_size,' 60')
coord_plot(180,-30-2,coord_color,coord_size,'  -30')
coord_plot(180,-60-3,coord_color,coord_size,'     -60')

plt.savefig('mollview_varing_lmax_TTEE_PECVEC_max1800_ticks_one_column_v3.pdf',dpi=600, bbox_inches = 'tight',pad_inches = 0)

