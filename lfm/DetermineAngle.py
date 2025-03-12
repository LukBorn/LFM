# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python [conda env:opm]
#     language: python
#     name: conda-env-opm-py
# ---

from matplotlib import pyplot as plt
# %matplotlib widget

# +

import serial
import numpy as np
import struct
from matplotlib import pyplot as plt

import time
import os
import h5py as h5
import pandas as pd
from skimage.feature import register_translation
import arrow
from scipy.ndimage import fourier_shift
from scipy.stats import linregress
def overlay(im0,im1,c0=20,c1=99):
    '''Magenta-Green Overlay Of Two Images'''
    #im0=(im0.astype('float')/np.max(im0))
    #im0=(im1.astype('float')/np.max(im1))
    i0=np.percentile(im0,c0)
    i1=np.percentile(im0,c1)
    im0=(im0-i0)/(i1-i0)
    i0=np.percentile(im1,c0)
    i1=np.percentile(im1,c1)
    im1=(im1-i0)/(i1-i0)
    im=np.zeros(im0.shape+(3,))
    im[:,:,0]=im0
    im[:,:,1]=im1
    im[:,:,2]=im0
    im[im>1]=1
    im[im<0]=0
    return im



# +
stage = serial.Serial(port='COM1', baudrate=9600, bytesize=8, parity='N', stopbits=1, timeout=30, xonxoff=False, rtscts=False, dsrdtr=False)

def getStatus():
    stage.write(b's\r') # send status command
    rrr = stage.read(32) # read return of 32 bytes without carriage return
    stage.read(1) # read and ignore the carriage return
    rrr
    statusbytes = struct.unpack(32*'B',rrr)
    resbytes=stage.inWaiting()
    stage.read(resbytes)
def getPosition():
    stage.write(b'c\r') # send commend to get position
    xyzb = stage.read(13) # read position from controller
    xyz_um = np.array(struct.unpack('lll', xyzb[:12]))/stepMult # convert bytes into 'signed long' numbers 
    resbytes=stage.inWaiting()
    stage.read(resbytes)
    return xyz_um
def setVelocity(vel):
    velb = struct.pack('H',int(vel))
    stage.write(b'V'+velb+b'\r')
    resbytes=stage.inWaiting()
    stage.read(resbytes)

def gotoPosition(pos):
    xyzr = struct.pack('lll',int(pos[0]*stepMult),int(pos[1]*stepMult),int(pos[2]*stepMult)) # convert integer values into bytes
    stage.write(b'm'+xyzr+b'\r') # send position to controller; add the "m" and the CR to create the move command
    cr = []
    cr = stage.read(1) # read carriage return and ignore
    resbytes=stage.inWaiting()
    stage.read(resbytes)
    
stepMult = 25
#Open H5 file and get zero position, determine measurement range
pos0=getPosition()
setVelocity(60)
# -

zz=np.arange(-50,50,10)
p=os.path.join("F:\\",arrow.now().format('YYYYMMDD_HHmm_')+'opm_calibration')
name='reg'

if not os.path.exists(p):
    os.mkdir(p)
    fh5=h5.File(p+"/data.h5")
    fh5.create_dataset("zz",data=zz)
else:
     fh5=h5.File(p+"/data.h5")


# #### Manual Measuring of the Angle

# Open GUI and Preview Bead Plane, Choose a bead on the camera surface that is in focus and register position p1

p1=getPosition()
thetas=[]
ps=[]

# for a bunch of position throughout the axial FOV bring the bead in focus with help the Z and Y controller of the stage and for every position execute to register it:

p2=getPosition()
ps.append(p2)
thetas.append(np.arctan((p2-p1)[0]/(p2-p1)[2]))
theta=thetas[-1]

ps=np.array(ps)
theta=np.nanmedian(np.arctan((ps[:,0]-ps[0,0])/(ps[:,2]-ps[0,2])))
ps_df=pd.DataFrame(ps)
ps_df2=ps_df-p1
ps_df2.to_csv(os.path.join(p,'angle_measurement.csv'))

fig,ax=plt.subplots()
res_lin=linregress(ps_df2[2],ps_df2[1])
ax.plot(ps_df2[2],ps_df2[1],'o')
ax.plot(ps_df2[2],ps_df2[2]*res_lin.slope+res_lin.intercept)
theta=np.arctan(res_lin.slope)*180/np.pi
ax.set_xlabel('axial stage displacement')
ax.set_ylabel('lateral stage displacement')
ax.annotate("angle in water in deg {:.2f}".format(np.arctan(res_lin.slope)*180/np.pi),[ps_df2[2].min(),ps_df2[1].min()])
plt.savefig(os.path.join(p,'angle_measurement.pdf'))
print(np.arctan(res_lin.slope)*180/np.pi)

gotoPosition(pos0)


mic=opm("./config.json","F:/")
name='x'
ims,conf,bg=mic.Acquire(name,25,10,10,0,-2.0,2.0,np.arange(-50,50,1),nFrames=1,todisk=False,bg=True)

fig,ax=plt.subplots()
ax.imshow(np.squeeze(ims.max(1)))

#shift specimen, take stack, and project
#shift specimen, take stack, and project
mips_1=[]
mips_2=[]
mips_3=[]
full=[]
for ii,iz in enumerate(zz):
    gotoPosition(pos0+[0,0,iz])
    time.sleep(1)
    ims,conf,bg=mic.Acquire(name,25,10,10,0,-2.0,2.0,np.arange(-50,50,1),nFrames=1,todisk=False,bg=True)
    mips_1.append(np.squeeze(ims.max(1)))
    mips_2.append(np.squeeze(ims.max(2)))
    mips_3.append(np.squeeze(ims.max(3)))
    del ims
mips_1=np.array(mips_1)
mips_2=np.array(mips_2)
mips_3=np.array(mips_3)
fh5.create_dataset('dz_mip_1',data=mips_1)
fh5.create_dataset('dz_mip_2',data=mips_2)
fh5.create_dataset('dz_mip_3',data=mips_3)


#Close and reopen file
import json
with open(p+"/para.json",'w') as f:
    json.dump(conf,f)
fh5.close()
fh5=h5.File(p+"/data.h5",'r')

# +
#FIND Z Pixel Pitch in xz MIP this is the distance between z planes
s='z'
ish=0
i_m=1
ds=np.copy(fh5[s+s])

mip=fh5['d'+s+'_mip_'+str(i_m)]
shifts=[]
for ii in range(len(ds)):
    im1=np.squeeze(mip[0,:,:])
    im2=np.squeeze(mip[ii,:,:])
    shift, error, diffphase = register_translation(im1, im2)
    shifts.append(shift)
shifts=np.array(shifts)

# +
dz,residual,rank,sinval=np.linalg.lstsq(np.vstack([shifts[:,ish],np.ones(len(shifts[:,ish]))]).T,ds,rcond=None)
fig,ax=plt.subplots()
plt.suptitle('Stage Displacement along '+s+' observed in XZ\'\'-Projection' )
ax.plot(np.copy(ds),shifts[:,0],'o',label='along z')
ax.plot(np.copy(ds),(ds-dz[1])/dz[0])
ax.plot(np.copy(ds),shifts[:,1],'o',label='along x')
ax.set_ylabel('Shifts in pixel on Camera')
ax.set_xlabel('Displacement in micron with stage')
plt.legend()

#ax[1].imshow(np.squeeze(np.single(mip[0,:,:])-mip[8,:,:]))
plt.savefig(p+'/'+s+'_displacement.pdf')

# +
#for a z displacment, find y' and z' displacment in pixels 
s='z'
i_m=3
ish=1
ds=np.copy(fh5[s+s])

mip=fh5['d'+s+'_mip_'+str(i_m)]
shifts=[]
for ii in range(len(ds)):
    im1=np.squeeze(mip[0,:,:])
    im2=np.squeeze(mip[ii,:,:])
    shift, error, diffphase = register_translation(im1, im2,upsample_factor=4)
    shifts.append(shift)
shifts=np.array(shifts)
shift_slope,residual,rank,sinval=np.linalg.lstsq(np.vstack([shifts[:,1],np.ones(len(shifts[:,1]))]).T,shifts[:,0],rcond=None)
dy,residual,rank,sinval=np.linalg.lstsq(np.vstack([ds,np.ones(len(ds))]).T,shifts[:,0],rcond=None)
fig,ax=plt.subplots()
plt.suptitle('Stage Displacement along '+s+' observed in YZ\'\'-Projection' )

ax.plot(np.copy(ds),shifts[:,0],'o',label='along y')
ax.plot(np.copy(ds),shifts[:,1],'o',label='along z')
ax.plot(np.copy(ds),ds*dy[0]+dy[1])
ax.set_ylabel('Shifts in pixel on Camera')
ax.set_xlabel('Displacement in micron with stage')
plt.legend()
#ax[1].imshow(np.squeeze(np.single(mip[0,:,:])-mip[-1,:,:]).T)
plt.savefig(p+'/'+s+'_displacement_yz.pdf')
# -

px0=2.5
print('Pixelsize at camera is  {:.2f} micron'.format(px0))
f_tl=45
f_obj1=20
fp_theta=35*np.pi/180
pl_theta=np.abs(theta)*np.pi/180
M1=f_tl/f_obj1
px1=px0/M1
print('Pixelsize at straight FP surface is  {:.2f} micron'.format(px1))
px2=px1/np.cos(fp_theta)
print('Pixelsize at oblique FP surface is  {:.2f} micron'.format(px2))
px3=px2*1.33
print('Pixelsize in final unsheared volume is')
print(px3)
#Nex pixelsizes
pxsz_sh = [1, px3*np.cos(pl_theta) ,px1]
print(pxsz_sh)


np.arctan(shift_slope[0]*1/1.69)*180/np.pi
