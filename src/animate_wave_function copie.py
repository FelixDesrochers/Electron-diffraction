#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.gridspec as gridspec
from WaveFunction_CN import WaveFunction
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

plt.rcParams.update({'font.size': 7})

################################################
#     1) Helper function for the simulation    #
################################################
# i) Gaussian wave packet
def gauss_xy(x, y, delta_x, delta_y, x0, y0, kx0, ky0):
    return 1/(2*delta_x**2*np.pi)**(1/4) * 1/(2*delta_y**2*np.pi)**(1/4) * np.exp(-((x-x0)/(2*delta_x)) ** 2) * np.exp(-((y-y0)/(2*delta_y)) ** 2) * np.exp( 1.j * (kx0*x + ky0*y))

# ii) Heaviside function for the square potential
def potential_heaviside(V0, x0, xf, y0, yf, x, y):
    V = np.zeros(len(x)*len(y))
    size_y = len(y)
    for i,yi in enumerate(y):
        for j,xj in enumerate(x):
            if (xj >= x0) and (xj <= xf) and (yi >= y0) and (yi <= yf):
                V[i+j*size_y] = V0
            else:
                V[i+j*size_y] = 0
    return V

# iii)
def intervalle(max_list,min_list,list_ref,n=3):
    return [round(i, -int(np.floor(np.log10(i))) + (n - 1))  for i in list_ref if (i < max_list) and (i > min_list) ]

# iv) Analytical model
def analytic_modulus(x, y, a, x0, y0, kx0, ky0, t):
    sigma = np.sqrt(a**2 + t**2/(4*a**2))
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-x0-(kx0)*t)/sigma)**2) * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2*((y-y0-(ky0)*t)/sigma)**2)

def compute_err(z1,z2,x,y):
    return np.trapz(np.trapz(abs(z1-z2), x).real, y).real

#####################################
#       2) Create the system        #
#####################################
# specify time steps and duration
dt = 0.005

# specify constants
hbar = 1.0   # planck's constant
m = 1.0      # particle mass

# specify range in x coordinate
x_min = -8
x_max = 13
dx = 0.08
x = np.arange(x_min, x_max+dx, dx)

# specify range in y coordinate
y_min = -12
y_max = 12
dy = dx
y = np.arange(y_min, y_max+dy, dy)

ni = 250
xi = np.linspace(x.min(),x.max(),ni)
yi = np.linspace(y.min(),y.max(),ni)
xig, yig = np.meshgrid(xi, yi)

#Create the potential
V0 = 400

x01 = 0
xf1 = 0.3
y01 = y.min()
yf1 = -2.85

#x0m = x01
#xfm = xf1
#y0m = -0.5
#yfm = 0.5

x02 = x01
xf2 = xf1
y02 = -yf1
yf2 = y.max()

V_xy = potential_heaviside(V0,x01,xf1,y01,yf1,x,y) + potential_heaviside(V0,x02,xf2,y02,yf2,x,y) #+ potential_heaviside(V0,x0m,xfm,y0m,yfm,x,y)

#V_xy = np.zeros(len(x)*len(y))

#Specify the parameter of the initial gaussian packet
x0 = -5
y0 = 0
#kx0 = 2*np.sqrt(11)
kx0 = 20
ky0 = 0
delta_x = 0.7
delta_y = 0.7

#Create the initial wave packet
size_x = len(x)
size_y = len(y)
xx, yy = np.meshgrid(x,y)
psi_0 = gauss_xy(xx, yy, delta_x, delta_y, x0, y0, kx0, ky0).transpose().reshape(size_x*size_y)

# Define the Schrodinger object which performs the calculations
S = WaveFunction(x=x, y=y, psi_0=psi_0, V=V_xy, dt=dt, hbar=hbar,m=m)
S.psi = S.psi/S.compute_norm()

######################################
#       3) Setting up the plot       #
######################################

#Setting up parameters for the
nb_frame = 300
nbr_level = 200

#Create the figure
fig = plt.figure(figsize=(11,8))
gs = gridspec.GridSpec(3, 3, width_ratios=[1,1,1.5], height_ratios=[1,0.1,1])
ax1 = plt.subplot(gs[:,:-1])
ax2 = plt.subplot(gs[0,-1],projection='3d')
ax3 = plt.subplot(gs[2,-1])
div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '3%', '3%')

#Aesthetic component of the graph
ax1.set_aspect(1)
ax1.set_xlim([x_min,x_max])
ax1.set_ylim([y_min,y_max])
ax1.set_xlabel(r"x ($a_0$)", fontsize = 16)
ax1.set_ylabel(r"y ($a_0$)", fontsize = 16)

ax2.view_init(elev=40., azim=-25.)
ax2.set_aspect(1)
ax2.set_xlim([x_min,x_max])
ax2.set_ylim([y_min,y_max])
ax2.set_xlabel(r"x ($a_0$)", fontsize = 9)
ax2.set_ylabel(r"y ($a_0$)", fontsize = 9)

ax3.set_xlim([y_min, y_max])
ax3.set_xlabel(r"y ($a_0$)", fontsize = 9)
ax3.set_ylabel(r"$|\psi(y,t)|^2$", fontsize = 9)

#Initial plotting
t = 0
z = S.get_prob().reshape(size_x,size_y).transpose()

level = np.linspace(0,z.max(),nbr_level)
cset = ax1.contourf(xx, yy, z, levels=level, cmap=plt.cm.jet, zorder=1)

#Draw the potential
ax1.text(0.02, 0.92, r"t = 0.0000  (u.a.)".format(S.t), color='white', transform=ax1.transAxes, fontsize=12)
ax1.vlines(x01, y01, yf1, colors='white', zorder=2)
ax1.vlines(xf1, y01, yf1, colors='white', zorder=2)
ax1.vlines(x02, y02, yf2, colors='white', zorder=2)
ax1.vlines(xf2, y02, yf2, colors='white', zorder=2)
ax1.hlines(yf1, x01, xf1, colors='white', zorder=2)
ax1.hlines(y02, x01, xf1, colors='white', zorder=2)
#ax1.hlines(y0m, x0m, xfm, colors='white', zorder=2)
#ax1.hlines(yfm, x0m, xfm, colors='white', zorder=2)
#ax1.vlines(x0m, y0m, yfm, colors='white', zorder=2)
#ax1.vlines(xfm, y0m, yfm, colors='white', zorder=2)

zi = griddata((xx.reshape(size_x*size_y), yy.reshape(size_x*size_y)), z.reshape(size_x*size_y), (xi[None,:], yi[:,None]), method='cubic')
ax2.plot_surface(xig, yig, zi, cmap=plt.cm.jet, rcount=ni, ccount=ni, alpha=0.95)
#ax2.grid(False)

#ax2.plot_surface(xx, yy, z, cmap=plt.cm.jet, zorder=1,rcount=75,ccount=75,antialiased=False)
z_i = 0.0
ax2.plot([x01,xf1,xf1,x01,x01], [y01,y01,yf1,yf1,y01], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)
ax2.plot([x02,xf2,xf2,x02,x02], [y02,y02,yf2,yf2,y02], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)
#ax2.plot([x0m,xfm,xfm,x0m,x0m], [y0m,y0m,yfm,yfm,y0m], z_i*np.ones(5), color='k', linewidth=2, zorder=2, alpha=1.)

#iii) third plot
x_desired = 11
k = abs(x-x_desired).argmin()
ax3.plot(yy[:,k],z[:,k])
#ax3.set_ylim([0, z[:,k].max()+0.01])
ax3.set_ylim([0, 0.23])
ax1.vlines(x[k], y_min, y_max, colors='orange', linestyle='dashed', zorder=2)

#Setting the colorbar
cbar1 = fig.colorbar(cset, cax=cax1)
major_ticks = np.linspace(0,4*z.max(),50)
ticks = intervalle(z.max(), 0, major_ticks)
cbar1.set_ticks(ticks)
cbar1.set_ticklabels(ticks)

t_vec = np.arange(0,nb_frame*dt,dt)
coupe = np.zeros((nb_frame,len(z[:,k])))

#Create animation
def animate(i):
    t = t_vec[i]
    S.step()
    z = S.get_prob().reshape(size_x,size_y).transpose()
    coupe[i] = z[:,k]

    ax1.clear()
    ax2.clear()
    ax3.clear()

    #plotting
    #i) first plot
    level = np.linspace(0,z.max(),nbr_level)
    cset = ax1.contourf(xx, yy, z, levels=level, cmap=plt.cm.jet,zorder=1)
    ax1.set_xlabel(r"x ($a_0$)", fontsize = 16)
    ax1.set_ylabel(r"y ($a_0$)", fontsize = 16)
    #ii) second plot
    zi = griddata((xx.reshape(size_x*size_y), yy.reshape(size_x*size_y)), z.reshape(size_x*size_y), (xi[None,:], yi[:,None]), method='cubic')
    ax2.plot_surface(xig, yig, zi, cmap=plt.cm.jet, rcount=ni, ccount=ni, alpha=0.95)
    ax2.set_zlim([0,zi.max()])
    ax2.set_xlabel(r"x ($a_0$)", fontsize = 9)
    ax2.set_ylabel(r"y ($a_0$)", fontsize = 9)
    ax2.set_xlim([x_min,x_max])
    ax2.set_ylim([y_min,y_max])
    #ax2.grid(False)
    #iii)third plot
    ax3.plot(yy[:,k],z[:,k])
    ax3.set_xlim([y_min, y_max])
    ax3.set_ylim([0, 0.23])
    ax3.set_xlabel(r"y ($a_0$)", fontsize = 9)
    ax3.set_ylabel(r"$|\psi(y,t)|^2$", fontsize = 9)

    #Draw the potential
    ax1.text(0.02, 0.92, r"t = {0:.3f} (u.a.)".format(S.t), color='white', transform=ax1.transAxes, fontsize=12)
    ax1.vlines(x01, y01, yf1, colors='white', zorder=2)
    ax1.vlines(xf1, y01, yf1, colors='white', zorder=2)
    ax1.vlines(x02, y02, yf2, colors='white', zorder=2)
    ax1.vlines(xf2, y02, yf2, colors='white', zorder=2)
    ax1.hlines(yf1, x01, xf1, colors='white', zorder=2)
    ax1.hlines(y02, x01, xf1, colors='white', zorder=2)
    #ax1.vlines(x0m, y0m, yfm, colors='white', zorder=2)
   # ax1.vlines(xfm, y0m, yfm, colors='white', zorder=2)
    #ax1.hlines(y0m, x0m, xfm, colors='white', zorder=2)
    #ax1.hlines(yfm, x0m, xfm, colors='white', zorder=2)
    ax2.plot([x01,xf1,xf1,x01,x01], [y01,y01,yf1,yf1,y01], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([x02,xf2,xf2,x02,x02], [y02,y02,yf2,yf2,y02], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    #ax2.plot([x0m,xfm,xfm,x0m,x0m], [y0m,y0m,yfm,yfm,y0m], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax1.vlines(x[k], y_min, y_max, colors='orange', linestyle='dashed', zorder=2)

    #Adjust the colorbar
    cbar1 = fig.colorbar(cset, cax=cax1)
    ticks = intervalle(z.max(), 0, major_ticks)
    cbar1.set_ticks(ticks)
    cbar1.set_ticklabels(ticks)

    print(i)


interval = 0.001
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
anim = animation.FuncAnimation(fig,animate,nb_frame,interval=interval*1e+3,blit=False)
anim.save('2D_2slit_dx={0}_dt={1}_yf1={2}_k={3}.mp4'.format(dx,dt,abs(yf1),kx0), fps=15, extra_args=['-vcodec', 'libx264'])

with open("2_slit_dx={0}_dt={1}_yf1={2}_k={3}.pkl".format(dx,dt,abs(yf1),kx0), 'wb') as pickleFile:
    pickle.dump(coupe, pickleFile)
    pickleFile.close()

exit()
plt.show()
