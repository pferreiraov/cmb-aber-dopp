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

#cmap_test = colors.LinearSegmentedColormap.from_list("", ["red","orange","green","blue"])
#cmap_func = matplotlib.cm.get_cmap(cmap_test)
lmin = 200
c=299792.458


vec_l = np.array([1000,1200,1400,1600,1800])
cmap_test = colors.LinearSegmentedColormap.from_list("t", ["red","orange","green","blue","purple"])
cmap_func = matplotlib.cm.get_cmap(cmap_test)

ab_vec_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[0] for i in vec_l])
dopp_vec_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[1] for i in vec_l])
abdopp_vec_nilc = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_nilc_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l])

ab_vec_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[0] for i in vec_l])
dopp_vec_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[1] for i in vec_l])
abdopp_vec_smica = np.array([np.loadtxt("data/48dir_ut78symm_2018_apod10_debeamed_noise_smica_w_DD_results_ab_dopp_abdopp_sim_estimator_clfrac_ABSUnbiasedVEC_lmin_"+str(lmin)+"_lmax_"+str(i)+".dat")[2] for i in vec_l])

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



for i in range(vec_l.size):
    disk_plot(vec_l[i],ab_vec_nilc[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],dopp_vec_nilc[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],abdopp_vec_nilc[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],ab_vec_smica[i],0.1,nside)
    
for i in range(vec_l.size):
    disk_plot(vec_l[i],dopp_vec_smica[i],0.1,nside)
    
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

laterror_smica = np.loadtxt('data/smicaTT1sigmaelipsevalues.dat')[:,1]
longerror_smica = np.loadtxt('data/smicaTT1sigmaelipsevalues.dat')[:,0]

laterror_smica2 = np.loadtxt('data/smicaTT2sigmaelipsevalues.dat')[:,1]
longerror_smica2 = np.loadtxt('data/smicaTT2sigmaelipsevalues.dat')[:,0]

aberror=np.array([laterror_smica[0],longerror_smica[0]]);
dopperror=np.array([laterror_smica[1],longerror_smica[1]]);
aberror2=np.array([laterror_smica2[0],longerror_smica2[0]]);
dopperror2=np.array([laterror_smica2[1],longerror_smica2[1]]);
#
abvec = ab_vec_smica[vec_l.size-1]
ablon,ablat=hp.vec2ang(abvec,lonlat=True)
doppvec = dopp_vec_smica[vec_l.size-1]
dopplon,dopplat=hp.vec2ang(doppvec,lonlat=True)
marker='v';colorm='b';#title='S';
ablongerror1=aberror[1];ablaterror1=aberror[0];colore1='grey';
ablongerror2=aberror2[1];ablaterror2=aberror2[0];colore2='lightgrey'
dopplongerror1=dopperror[1];dopplaterror1=dopperror[0];colore1='grey';
dopplongerror2=dopperror2[1];dopplaterror2=dopperror2[0];colore2='lightgrey'
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
        
def error_bar_dopp(lat_start,long_start,width,headwidth,headlength,r2,r1,r22s,r12s,ticks=False):
#   Setting the projection
#    proj = hp.projector.MollweideProj()
#    hp.visufunc.projplot(lon,lat,'^',lonlat=True,color='y', markersize=4)
    if ticks==True:
        hp.visufunc.projplot(long_start+r1*np.cos(phi),lat_start + (r2)*np.sin(phi),lonlat=True,color='purple',linestyle='dashed',linewidth=2)
#        hp.visufunc.projplot(long_start+r12s*np.cos(phi),lat_start + (r22s)*np.sin(phi),lonlat=True,color='plum',linestyle='dashed',linewidth=2)    
    else:
        hp.visufunc.projplot(long_start+r1*np.cos(phi),lat_start + (r2)*np.sin(phi),lonlat=True,color='purple')
#        hp.visufunc.projplot(long_start+r12s*np.cos(phi),lat_start + (r22s)*np.sin(phi),lonlat=True,color='plum')
#   Plotting vectors form (lat_start,long_start) coordinates to (lat_end,long_end) coordinates
#    plt.annotate('', xy=(proj.ang2xy(lat_end, long_end,lonlat=True)), xytext=(proj.ang2xy(lat_start, long_start,lonlat=True)),   arrowprops=dict(color=color,width=width,headwidth=headwidth,headlength=headlength))


error_bar(ablat,ablon,0.5,6,6,ablaterror1,ablongerror1,ablaterror2,ablongerror2)

error_bar_dopp(dopplat,dopplon,0.5,6,6,dopplaterror1,dopplongerror1,dopplaterror2,dopplongerror2)

laterror_nilc = np.loadtxt('data/nilcTT1sigmaelipsevalues.dat')[:,1]
longerror_nilc = np.loadtxt('data/nilcTT1sigmaelipsevalues.dat')[:,0]

laterror_nilc2 = np.loadtxt('data/nilcTT2sigmaelipsevalues.dat')[:,1]
longerror_nilc2 = np.loadtxt('data/nilcTT2sigmaelipsevalues.dat')[:,0]

aberror=np.array([laterror_nilc[0],longerror_nilc[0]]);
dopperror=np.array([laterror_nilc[1],longerror_nilc[1]]);
aberror2=np.array([laterror_nilc2[0],longerror_nilc2[0]]);
dopperror2=np.array([laterror_nilc2[1],longerror_nilc2[1]]);
#
abvec = ab_vec_nilc[vec_l.size-1]
ablon,ablat=hp.vec2ang(abvec,lonlat=True)
doppvec = dopp_vec_nilc[4]
dopplon,dopplat=hp.vec2ang(doppvec,lonlat=True)
marker='v';colorm='b';#title='S';
ablongerror1=aberror[1];ablaterror1=aberror[0];colore1='grey';
ablongerror2=aberror2[1];ablaterror2=aberror2[0];colore2='lightgrey'
dopplongerror1=dopperror[1];dopplaterror1=dopperror[0];colore1='grey';
dopplongerror2=dopperror2[1];dopplaterror2=dopperror2[0];colore2='lightgrey'
#
error_bar(ablat,ablon,0.5,6,6,ablaterror1,ablongerror1,ablaterror2,ablongerror2,ticks=True)

error_bar_dopp(dopplat,dopplon,0.5,6,6,dopplaterror1,dopplongerror1,dopplaterror2,dopplongerror2,ticks=True)

#error_bar(0,0,0.5,6,6,30,50,ticks=False)

########################################
########################################

for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],ab_vec_nilc[i],11.5,'s',fill_style='left')
    
for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],dopp_vec_nilc[i],11.5,'o',fill_style='left')
    
#for i in range(6):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],abdopp_vec_nilc[i],6,'D',fill_style='left')
    
for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],ab_vec_smica[i],11.5,'s',fill_style='right')
    
for i in range(vec_l.size):
    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],dopp_vec_smica[i],11.5,'o',fill_style='right')

dipole_lat = 48.253
dipole_long = 264.021
dipole_beta = 0.00123357
fiducialvec = hp.ang2vec(dipole_long,dipole_lat,lonlat=True)*dipole_beta   
hp.visufunc.projplot(dipole_long,dipole_lat,'*',lonlat=True,color='k',markersize=12.5)
    
#for i in range(6):
#    mollweid_vec(vec_l[i],vec_l[0],vec_l[-1],abdopp_vec_smica[i],6,'D',fill_style='right')
plt.rcParams["legend.fontsize"] = 15.3
legend1 = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
                          markersize=15, label='',fillstyle='right')
legend2 = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
                          markersize=15, label='',fillstyle='right')
#legend3 = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
#                          markersize=10, label='',fillstyle='right')
legend3 = mlines.Line2D([], [], color='k', marker='s', linestyle='None',
                          markersize=15, label='Aberration',fillstyle='left')
legend4 = mlines.Line2D([], [], color='k', marker='o', linestyle='None',
                          markersize=15, label='Doppler',fillstyle='left')
#legend6 = mlines.Line2D([], [], color='k', marker='D', linestyle='None',
#                          markersize=10, label='Pec. Velocity',fillstyle='left')
fig = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
norm = matplotlib.colors.BoundaryNorm(vec_l, cmap_test.N)
cmap = fig.colorbar(image,orientation='horizontal',fraction=0.025,pad=0.01,aspect=50, ticks=vec_l)#,spacing='proportional', ticks=vec_l, boundaries=vec_l, format='%1i')
cmap.ax.tick_params(labelsize=18)

ax.text(-0.09, -1.38,r'$\ell_{max}$', fontsize=22)
ax.text(-2.0, 0.78, 'T', fontsize=30)
ax.text(-1.856, 0.78, 'T', fontsize=30)

#ax.text(0.35, 0.222,'Doppler', fontsize=16,color='grey')
#ax.text(0.35, -0.148,'Doppler', fontsize=16,color='lightgrey')

#ax.text(1.2, 0.34,'Aberration', fontsize=16,color='grey')
#ax.text(1.3, 0.13,'Aberration', fontsize=16,color='lightgrey')

plt.legend(handles=[legend1, legend2, legend3, legend4],loc="lower right",title="SMICA   NILC", title_fontsize=15.3, ncol=2,handletextpad=1.24)._legend_box.align='left'

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

dipole_lat = 48.253
dipole_long = 264.021
dipole_beta = 0.00123357
fiducialdirvec= hp.ang2vec(dipole_long,dipole_lat,lonlat=True)
fiducialvec = hp.ang2vec(dipole_long,dipole_lat,lonlat=True)*dipole_beta
fiducialvecnorm = np.linalg.norm(fiducialvec)
plt.savefig('mollview_varing_lmax_TT_AB+DOPP_max1800_ticks_one_column.pdf',dpi=600, bbox_inches = 'tight',pad_inches = 0)

