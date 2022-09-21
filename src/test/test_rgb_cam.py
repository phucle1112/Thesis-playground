print ('First test on remote coding on Pi')
print ('Test processing images on remote VSC on Pi')

import time
from picamera import PiCamera
import numpy as np
import matplotlib.pyplot as plt

# cam = PiCamera()
# cam.resolution = (2560, 1936)
# cam.start_preview()
# sleep(5)
# cam.capture()
# cam.stop_preview()

t1 = time.time()
h = 640 # change this to anything < 2592 (anything over 2000 will likely get a memory error when plotting
cam_res = (int(h),int(0.75*h)) # keeping the natural 3/4 resolution of the camera
cam_res = (int(16*np.floor(cam_res[1]/16)),int(32*np.floor(cam_res[0]/32)))
cam = PiCamera()
cam.resolution = (cam_res[1],cam_res[0])
data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8)
x,y = np.meshgrid(np.arange(np.shape(data)[1]),np.arange(0,np.shape(data)[0]))
cam.capture(data,'rgb')
fig,axn = plt.subplots(2,2,sharex=True,sharey=True)
fig.set_size_inches(9,7)
axn[0,0].pcolormesh(x,y,data[:,:,0],cmap='hot')
axn[0,1].pcolormesh(x,y,data[:,:,1],cmap='hot')
axn[1,0].pcolormesh(x,y,data[:,:,2],cmap='hot')
axn[1,1].imshow(data,label='Full Color')
axn[0,0].title.set_text('Red')
axn[0,1].title.set_text('Green')
axn[1,0].title.set_text('Blue')
axn[1,1].title.set_text('All')
print('Time to plot all images: {0:2.1f}'.format(time.time()-t1))
data = np.empty((cam_res[0],cam_res[1],3),dtype=np.uint8)
plt.savefig('component_plot.png',dpi=150)
plt.cla()
plt.close()