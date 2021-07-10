# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:51:19 2021

@author: Fardin
"""
# Imports(Modules to use)
import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm


# %% Functions

#  This is the advection reaction diffusion solver.
def ard_pde(up,te,wp,du,dte,dw,dt,dx,dy,k1,tem,salt_conc):
# This numerically integrates using finite differences and the forward
# Euler method.
        
    # Set up zero flux bc's in x.
    # These pad the array with the second and second to last items,
    # Thus the last item has the same element on both sides.
    upy = np.column_stack((up[:,1],up,up[:,-2]))
    tey = np.column_stack((te[:,1],te,te[:,-2]))
    wpy = np.column_stack((wp[:,1],wp,wp[:,-2]))
    # Make boundaries zero flux in y.
    upxy = np.vstack((upy[1,:], upy, upy[-2,:]))
    texy = np.vstack((tey[1,:], tey, tey[-2,:]))
    wpxy = np.vstack((wpy[1,:], wpy, wpy[-2,:]))
    #dum matrix method
    dum=du/(1+wpxy)
    # Perform finite differences. (Diffusion)
    # On axis terms for u
    # Calculate differences
    uxx=dt*(1/dx**2)*(dum[:,2:]*upxy[:,2:]+dum[:,0:-2]*upxy[:,0:-2]-2*dum[:,1:-1]*upxy[:,1:-1])
    uxx=uxx[1:-1,:] # Remove extra rows
    # Calculate differences
    uyy=dt*(1/dy**2)*(dum[2:,:]*upxy[2:,:]+dum[0:-2,:]*upxy[0:-2,:]-2*dum[1:-1,:]*upxy[1:-1,:])
    uyy=uyy[:,1:-1] # Remove extra columns
    #Diagnoal difference take into account diffusion barrier  
    # Perform finite differences. (Diffusion)
    # On axis terms for u
    # Calculate differences
    #uxx=dt*(1/dx**2)*(du*(1-wpxy[:,1:-1]))*( (1-wpxy[:,2:])*(upxy[:,2:]-upxy[:,1:-1])+(1-wpxy[:,0:-2])*(upxy[:,0:-2]-upxy[:,1:-1]))
    #uxx=uxx[1:-1,:] # Remove extra rows
    # Calculate differences
    #uyy=dt*(1/dy**2)*(du*(1-wpxy[1:-1,:]))*( (1-wpxy[2:,:])*(upxy[2:,:]-upxy[1:-1,:])+(1-wpxy[0:-2,:])*(upxy[0:-2,:]-upxy[1:-1,:]))
    #uyy=uyy[:,1:-1] # Remove extra columns
    # On axis terms for t
    # Calculate differences
    txx=dt*(dte/dx**2)*(texy[:,2:]+texy[:,0:-2]-2*texy[:,1:-1])
    txx=txx[1:-1,:] # Remove extra rows
    # Calculate differences
    tyy=dt*(dte/dy**2)*(texy[2:,:]+texy[0:-2,:]-2*texy[1:-1,:])
    tyy=tyy[:,1:-1] # Remove extra columns    
    # The included fudge-factor ff rounds out the square pixels
    # during diffusion. This factor downplays diagonal diffusion.
    ff=15;  # Set ff=1 to turn the fudge-factor off.
    #dum matrix method
    uxy=dt*((1/(dx**2+dy**2))*(dum[2:,2:]*upxy[2:,2:]+dum[0:-2,0:-2]*upxy[0:-2,0:-2]-2*dum[1:-1,1:-1]*upxy[1:-1,1:-1]))/ff
    uyx=dt*((1/(dx**2+dy**2))*(dum[0:-2,2:]*upxy[0:-2,2:]+dum[2:,0:-2]*upxy[2:,0:-2]-2*dum[1:-1,1:-1]*upxy[1:-1,1:-1]))/ff
    
    # Diagonal terms for u
    # Calculate differences
    #uxy=dt*((1/(dx**2+dy**2))*(du*(1-wpxy[1:-1,1:-1]))*( (1-wpxy[2:,2:])*(upxy[2:,2:]-upxy[1:-1,1:-1])+(1-wpxy[0:-2,0:-2])*(upxy[0:-2,0:-2]-upxy[1:-1,1:-1])))/ff
    #uyx=dt*((1/(dx**2+dy**2))*(du*(1-wpxy[1:-1,1:-1]))*( (1-wpxy[0:-2,2:])*(upxy[0:-2,2:]-upxy[1:-1,1:-1])+(1-wpxy[2:,0:-2])*(upxy[2:,0:-2]-upxy[1:-1,1:-1])))/ff
    # Diagonal terms for t
    # Calculate differences
    txy=dt*((dte/(dx**2+dy**2))*(texy[2:,2:]+texy[0:-2,0:-2]-2*texy[1:-1,1:-1]))/ff
    tyx=dt*((dte/(dx**2+dy**2))*(texy[0:-2,2:]+texy[2:,0:-2]-2*texy[1:-1,1:-1]))/ff
    
    # Combine diffusion with advection and kinetics using
    # the forward Euler algorithm.
    up=up+(uxx+uyy+uxy+uyx)+dt*(k1*up*te)
    te=te+(txx+tyy+txy+tyx)
    #For the ice creation
    x=0
    y=0
    
    for x in range(len(w[:])):
                   for y in range(len(w[:])):
                           if (te[x,y]<tem and up[x,y]>salt_conc):
                               wp[x,y]=1
                           else:
                               wp[x,y]=0
    #if (te[:,:]<0 and up[:,:]>0.8):
        #wp[:,:]=1
    #else:
        #wp[:,:]=0         
                        
                 
    # Here you apply any constant conditions
    # Holding the top left edge at a constant value
    tl_u = 0.0
    tl_t = 0.0
    # From fraction to end (top)
    up[0,int(np.round(5*ny/10)):] = tl_u
    te[0,int(np.round(5*ny/10)):] = tl_t
    # Holding the bottom left edge at a constant value
    bl_u = 0.0
    bl_t = 0.0
    # From begining (bottom) to fraction
    up[0,0:int(np.round(5*ny/10))] = bl_u
    te[0,0:int(np.round(5*ny/10))] = bl_t
    #Top Center(Constant Piece of Brine at the Top)
    tc_u=2.0
    spotwidth=2 # This is half width in steps
    spotleft=int(np.round(nx/2))-spotwidth   # Determine the left edge
    spotright=int(np.round(nx/2))+spotwidth  # Determine the right edge
    up[spotleft:spotright,-1] = tc_u
    # Top Center Temperature(For the intial brine)
    tc_t=-5.0
    spotwidth=2 # This is half width in steps
    spotleft=int(np.round(nx/2))-spotwidth   # Determine the left edge
    spotright=int(np.round(nx/2))+spotwidth  # Determine the right edge
    te[spotleft:spotright,-1] = tc_t 
    
        
    return [up, te, wp]
    
#define length of ice
def icelength(wp,t):
    w_length= np.sum(w[:,:,t],axis=0)
    w_nonzeros=np.flatnonzero(w_length)
    w_end=ny-w_nonzeros[0]
    return w_end/100



# %% Model Parameters

# The time and space parameters for the model go here.
res = 0.1       # This sets the resolution for the simulation in mm/step.
Lx = 50       # This sets the x length of the simulation in mm.
Ly = 50          # This sets the y length of the simulation in mm.
tEnd = 200  # This sets the duration of the simulation in s.
dt = 0.5        # The time step for the calculation in s.
dtWindow =5   # This sets how often to update the plot in s.

# These are internal parameters for the simulation.
dx = res                           # The x resolution for the calculation.
dy = res                           # The y resolution for the calculation.
nx = math.floor(Lx/dx)             # The x dimension for storage.
ny = math.floor(Ly/dy)             # The y dimension for storage.
nt = math.floor(tEnd/dt)+1         # The t dimension for storage.
u = np.zeros((nx,ny,nt))           # Define the array for u data.
te = np.zeros((nx,ny,nt))           # Define the array for t data.
w = np.zeros((nx,ny,nt))           # Define the array for w data.
ill=np.zeros((2,nt-1))                   # Final values of ice length stored in the list  

# %% Chemical Parameters

# The chemistry parameters of the model go here.
du = 0.02    # This sets the diffusion for u in mm^2/s.
dte = 0.02    # This sets the diffusion for v in mm^2/s.(for temperature)
dw = 0.000    # This sets the diffusion for v in mm^2/s.
#xvel = 0.0    # This set the x advection velocity in mm/s.
#yvel = 1.0    # This set the y advection velocity in mm/s.
#vel_u = 0     # This adjusts the advection velocity for u.
#vel_v = 0.1   # This adjusts the advection velocity for t.
#vel_w = 0     # This adjusts the advection velocity for w.
k1= 100     # Rate for the loss of salt water  in 1/(M*s).
salt_conc= 0.5      # Concentration of seawater at which ice forms
tem=0                 #temperature when ice forms
#k2= 0.0        # Rate of the loss in brine as Ice forms
#k3= 0.15     # Rate of the gain in brine as falling saltwater comes into contact with warmer water 
#k4= 0.55      # Rate of the formation of ice due to seoeration of saltwater   

# %% Initial Conditions
#Parameters:
    #u=saltwater
    #t=temperature
    #w=ice

# This sets the initial conditions for the simulation. Initially every spot
# is set to the same values with some optional random variational
# of amplitude amp.
u0 = 0        # Arbitrary initial value
v0 = 0        # Arbitrary initial value
w0 = 0        # Arbitrary initial value
ampu = 0      # Set to zero to start everything the same
ampv = 0      # Set to zero to start everything the same
ampw = 0      # Set to zero to start everything the same
u[:,:,0] = 0 #u0 + ampu * np.random.rand(nx,ny) # Adds noise to IC
te[:,:,0] = 0 #v0 + ampv * np.random.rand(nx,ny) # Adds noise to IC
w[:,:,0] = 0 #w0 + ampw * np.random.rand(nx,ny) # Adds noise to IC
uC = 0.5      # The u value for the different spots.
tC = 0.5      # The v value for the different spots.
wC = 0.5      # The v value for the different spots.

# This is the middle spot of specified width
w[0:nx,ny-2:ny,0]=1
spotwidth=2 # This is half width in steps
spotleft=int(np.round(nx/2))-spotwidth   # Determine the left edge
spotright=int(np.round(nx/2))+spotwidth  # Determine the right edge
# spottop=int(np.round(ny/2))-spotwidth    # Determine the top edge
# spotbottom=int(np.round(ny/2))+spotwidth # Determine the bottom edge
# u[spotleft:spotright,spottop:spotbottom,0]=uC # Create the initial spot in u
#v[spotleft:spotright,ny-5:ny-2,0]=2 # Create the initial spot in v
u[:,:,0]=0.1 #Intial concentration of seawater
# w[spotleft:spotright,spottop:spotbottom,0]=wC # Create the initial spot in w

# %% This section runs the simulation itself.
t = 0        # This sets the first time point for the calculation.
telap = 0    # This sets the time elapsed for the simulation.
s=[]
while telap < tEnd:    
       
    # From here on down is the nuts and bolts of the simulation.
    # Update u, v, and w for the next time output
    # using the pde function defined at the beginning of this file
    [u[:,:,t+1],te[:,:,t+1],w[:,:,t+1],]=ard_pde(u[:,:,t],te[:,:,t],w[:,:,t],du,dte,dw,dt,dx,dy,tem,salt_conc,k1)
    ill[0,t]=(t)*dt
    #ill[1,t]=icelength(w,t)
    ups=u[1:4,1:4,t+1] # Samples for debugging
    tps=te[1:4,1:4,t+1] # Samples for debugging
    wps=w[1:4,1:4,t+1] # Samples for debugging
    t=t+1              # Increment the storage counter
    telap=t*dt         # Increment the simulation time elapsed
    # This displays the step and time being plotted. 
    print('Step: {0} Time: {1:0.3f}s'.format(t,telap))

# %% Create the output files
    
files = []   # This is for the output files.
t = 0        # This sets the first time point for the calculation.
telap = 0    # This sets the time elapsed for the simulation.
while telap <= tEnd:    

    # Make the plots
    x = np.arange(0,Lx,dx) # Create the x data for plotting
    y = np.arange(0,Ly,dy) # Create the y data for plotting
    time=np.arange(0,tEnd+0.1,dt)    
    Y,X = np.meshgrid(y,x) # Create the X,Y matrices for plotting
    Z1 = u[:,:,t] # Pull the Z data matrix for plotting
    Z2 = te[:,:,t] # Pull the Z data matrix for plotting
    Z3 = w[:,:,t] # Pull the Z data matrix for plotting     
    Z4 = (u[:,:,t] + te[:,:,t]) # Pull the Z data matrix for plotting
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2) # Create the figure with subplots
    # Create the filled countour plot with colormap and manual levels
    cf1 = ax1.contourf(X,Y,Z1,cmap=cm.coolwarm,levels=np.arange(0,1.1,0.01))
    fig.colorbar(cf1, ax=ax1) # Add the colorbar
    ax1.set_xlabel('x (mm)') # Label the x axis
    ax1.set_ylabel('y (mm)') # Label the y axis
    utitle = 'Concentration of Saltwater \n at {0:0.1f}s'.format(telap)
    ax1.set_title(utitle) # Title the plot
    ax1.set_aspect('equal') # Make the aspect ratio equal
    # Create the filled countour plot with colormap and manual levels
    cf2 = ax2.contourf(X,Y,Z2,cmap=cm.coolwarm,levels=np.arange(0,1.1,0.01))
    fig.colorbar(cf2, ax=ax2) # Add the colorbar
    ax2.set_xlabel('x (mm)') # Label the x axis
    ax2.set_ylabel('y (mm)') # Label the y axis
    vtitle = 'Concentration of Temperature \n at {0:0.1f}s'.format(telap)
    ax2.set_title(vtitle) # Title the plot
    ax2.set_aspect('equal') # Make the aspect ratio equal
    # Create the filled countour plot with colormap and manual levels
    cf3 = ax3.contourf (X,Y,Z3,cmap=cm.coolwarm,levels=np.arange(0,1.1,0.01))
    fig.colorbar(cf3, ax=ax3) # Add the colorbar
    ax3.set_xlabel('x (mm)') # Label the x axis
    ax3.set_ylabel('y (mm)') # Label the y axis
    wtitle = 'Concentration of Ice \n at {0:0.1f}s'.format(telap)
    ax3.set_title(wtitle) # Title the plot
    ax3.set_aspect('equal') # Make the aspect ratio equal
     # Create the filled countour plot with colormap and manual levels
    #cf4 = ax4.contourf(X,Y,Z4,cmap=cm.coolwarm,levels=np.arange(0,2.2,0.01))
    #fig.colorbar(cf4, ax=ax4) # Add the colorbar
    #ax4.set_xlabel('x (mm)') # Label the x axis
    #ax4.set_ylabel('y (mm)') # Label the y axis
    #utitle = 'Concentration of u+v \n at {0:0.1f}s'.format(telap)
    #ax4.set_title(utitle) # Title the plot
    #ax4.set_aspect('equal') # Make the aspect ratio equal
    #Length
    #length_plot=ax4.plot(time[0:t],ill[1,0:t])
    length_plot1=ax4.plot(time[0:t],np.sqrt(ill[0,0:t]))
    ax4.set_xlabel('x (s)') # Label the x axis
    ax4.set_ylabel('y (cm)') # Label the y axis
    utitle = 'Length of Ice(Model and Theoretical)'
    ax4.set_title(utitle) # Title the plot
    
    # plt.subplots_adjust(hspace=0.75,left=-0.05)
    plt.tight_layout()
    # plt.subplots_adjust(left=-0.3)

    # plt.show() # This shows the plots as the code is running
    fname = 'Brinicles_%06d.png' % t # Create the file name for each plot
    print('Saving frame', fname) # Print the status update
    fig.savefig(fname, dpi=300) # Save the image
    files.append(fname) # Update the filename
    plt.close(fig) # Close the image so it doesn't show while the code is running
    plt.clf() # Clear the figrue to save memory
    t=t+math.floor(dtWindow/dt)            # Increment the storage counter
    telap=t*dt         # Increment the simulation time elapsed   
