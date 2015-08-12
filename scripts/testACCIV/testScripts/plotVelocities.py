#!/usr/bin/python
import numpy as np
import h5py

from optparse import OptionParser
import os.path
  
import matplotlib as mpl


parser = OptionParser()
parser.add_option("--folder", type="string", default='full/pass1/', dest="folder", help="folder of the output data to be plotted")
parser.add_option("--imageFileName", type="string", default='image001.h5', dest="imageFileName", help="folder of the output data to be plotted")
parser.add_option("--savePlots", action="store_true", dest="savePlots", help="include this flag save plots to files instead of displaying them")
parser.add_option("--gridFileName", type="string", default='outGridVelocity.h5', dest="gridFileName")
parser.add_option("--scatterFileName", type="string", default='outScatteredVelocity.h5', dest="scatterFileName")
parser.add_option("--figurePrefix", type="string", default='fig', dest="figurePrefix")
parser.add_option("--figureExt", type="string", default='.pdf', dest="figureExt")
parser.add_option("--tiePointsFolder", type="string", default='_work', dest="tiePointsFolder")


options, args = parser.parse_args()



if options.savePlots:
  mpl.use('Agg')
  
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from   mpl_toolkits.axes_grid1 import make_axes_locatable

folder = options.folder
scatterFileName = '%s/%s'%(folder,options.scatterFileName)
if(not os.path.exists(scatterFileName)):
  #print("not found:", scatterFileName)
  exit()
gridFileName = '%s/%s'%(folder,options.gridFileName)
if(not os.path.exists(gridFileName)):
  print("not found:", gridFileName)
  exit()
  
tiePointsFileName = '%s/%s/combinedCorrelationTiePoints.h5'%(folder,options.tiePointsFolder)
imageFileName = options.imageFileName
if(not os.path.exists(imageFileName)):
  print("not found:", imageFileName)
  exit()

# background
axisBG = [0.1,0.1,0.1]

# velocity norm range ( [xmin,ymin],[xmax,ymax])
velocityRange = np.array([[0,-0.4],[-2.0,0.4]])
velocityMaxNorm =  np.around(np.linalg.norm(velocityRange[1] - velocityRange[0]),decimals=1)


# plot a velocity vector every "skip" pixels from the gridded velocity data
skip = 6

# width and height of each figure (inches)
width = 12
height = 10

# number of points to be sampled from the scattered data
maxPoints = 10000

# the locations of the major and minor axes to plot
x0 = 0.05
y0 = 0.08
# the width around each axis to take points from when plotting axes
dx = 0.01
dy = 0.01

maxImageStd = 3.0 # Image data is clamped to be within 3 std. dev. from the mean.
                  # Decrease this number to increase image contrast.

# decrease these numbers to increase the length of vectors, and visa versa
scatterVectorScale = 200.0
gridVectorScale = 200.0

quiverOpts = {'headwidth': 2, 'headlength':4}

h5File = h5py.File(imageFileName, 'r')
bounds = h5File["bounds"][...]
imageData = h5File["data"][...]
print("Index (0,0) (top-left):", imageData[0,0])
print("Index (1,0):", imageData[1,0])
    
imageMask = np.array(h5File["mask"][...],bool)
imageVelocityMask = np.ma.masked_equal(h5File["velocityMask"][...],0)
h5File.close()


h5File = h5py.File(scatterFileName, 'r')
x = h5File["x"][...]
y = h5File["y"][...]
vx = h5File["vx"][...]
vy = h5File["vy"][...]
pixelVx = h5File["dataX"][...]
pixelVy = h5File["dataY"][...]
h5File.close()


h5File = h5py.File(tiePointsFileName, 'r')
deltaTs = h5File["deltaTs"][...]
residualsFound = "correlationVelocityResiduals" in h5File
if residualsFound:
  correlationVelocityResiduals = h5File["correlationVelocityResiduals"][...]
  correlationLocationResiduals = h5File["correlationLocationResiduals"][...]
h5File.close()

maxDeltaT = np.amax(deltaTs)

#print np.amax(np.abs(vx))
#print np.amax(np.abs(vy))

h5File = h5py.File(gridFileName, 'r')
gridVx = h5File["vx"][...]
gridVy = h5File["vy"][...]
h5File.close()
gx = np.linspace(bounds[0],bounds[1],gridVx.shape[1])
gy = np.linspace(bounds[2],bounds[3],gridVx.shape[0])



# File info: ===========================================================
print( "Grid Data size: " + str(gridVx.shape) )
print( "Scattered data size: " + str(vx.shape) )
print( "Scattered pixelVx size: " + str(pixelVx.shape) )
# ======================================================================

# Setup Data============================================================
dLon = gx[1]-gx[0]
dLat = gy[1]-gy[0]

pixelVx = pixelVx/dLon*maxDeltaT
pixelVy = pixelVy/dLat*maxDeltaT

[gridX, gridY] = np.meshgrid(gx,gy)

vMagGrid = np.sqrt(gridVx**2 + gridVy**2)
vMag = np.sqrt(vx**2+vy**2)

imageMean = np.mean(imageData[imageMask])
imageStd = np.std(imageData[imageMask])

imageData *= imageMask
#imageData = np.maximum(imageMean-maxImageStd*imageStd,
   #np.minimum(imageMean+maxImageStd*imageStd,imageData))

if(x.size > maxPoints):
  indices = np.array(np.random.rand(maxPoints)*x.size,int)
else:
  indices = np.array(np.linspace(0,x.size-1,x.size),int)

colorList1 = np.array(((0.0,0.0,0.0),
	(1.0,0.0,0.0),
	(1.0,1.0,0.0),
	(1.0,1.0,1.0)))

colorList2 = np.array(((0.5,0.5,0.5),
	(0.67,0.67,0.67),
	(0.83,0.83,0.83),
	(1.0,1.0,1.0)))
alpha = 0.25

colorList = alpha*colorList1 + (1-alpha)*colorList2
colorList = tuple(tuple(x) for x in colorList)


colormap = cm.Greys_r
#colormap = colors.LinearSegmentedColormap.from_list('my_map',colorList,N=256)


maskXAxis = np.abs(y-y0) < dy
maskYAxis = np.abs(x-x0) < dx
xAxisIndices = indices[maskXAxis[indices]]
yAxisIndices = indices[maskYAxis[indices]]

xAxisGridIndex = np.argmin(np.abs(gy-y0))
yAxisGridIndex = np.argmin(np.abs(gx-x0))


# Plot Data ============================================================
figCount = 1

fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal')
ax.imshow(imageData, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=colormap)
ax.quiver(x[indices], y[indices], vx[indices], vy[indices], color='g', pivot='tail',  scale_units='xy', scale=scatterVectorScale, **quiverOpts)
ax.quiver(x[xAxisIndices], y[xAxisIndices], vx[xAxisIndices], vy[xAxisIndices], color='r', pivot='mid',  scale_units='xy', scale=scatterVectorScale, **quiverOpts)
ax.quiver(x[yAxisIndices], y[yAxisIndices], vx[yAxisIndices], vy[yAxisIndices], color='b', pivot='mid',  scale_units='xy', scale=scatterVectorScale, **quiverOpts)
ax.set_title('a sample of %i scattered velocity vectors'%(indices.size))
ax.set_xlabel("$x$ $[m]$")
ax.set_ylabel("$y$ $[m]$")


##fig = plt.figure(12, figsize=[width,height])
###fig.subplots_adjust(left=0.075, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.25)
##ax = fig.add_subplot(111, aspect='equal')
##plt.imshow(imageData, extent=(bounds[0],bounds[1],bounds[3],bounds[2]), cmap=colormap)
##ax.set_ylim(ax.get_ylim()[::-1])
##plt.axis('tight')

figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal')
ax.imshow(imageData, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=colormap)
ax.quiver(gridX[::skip,::skip], gridY[::skip,::skip], gridVx[::skip,::skip], gridVy[::skip,::skip], color='g', pivot='tail',  scale_units='xy', scale=gridVectorScale)
ax.set_title('gridded velocity vector (skip = %i)'%skip)
ax.set_xlabel("$x$ $[m]$")
ax.set_ylabel("$y$ $[m]$")
fig.set_tight_layout("true")

figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal')
im = ax.imshow(gridVx, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=plt.get_cmap('jet'))
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="2%", pad=0.2)
fig.colorbar(im, cax = cax1)
im.set_clim(0,velocityMaxNorm)
ax.set_aspect('equal')
im.set_clim(velocityRange[0][0],velocityRange[1][0])
ax.set_title('$v_x$')
ax.set_xlabel("$x$ $[m]$")
ax.set_ylabel("$y$ $[m]$")
fig.set_tight_layout("true")

figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal')
im = ax.imshow(gridVy, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=plt.get_cmap('jet'))
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="2%", pad=0.2)
fig.colorbar(im, cax = cax1)
im.set_clim(0,velocityMaxNorm)
ax.set_aspect('equal')
im.set_clim(velocityRange[0][1],velocityRange[1][1])
ax.set_title('$v_y$')
ax.set_xlabel("$x$ $[m]$")
ax.set_ylabel("$y$ $[m]$")
fig.set_tight_layout("true")


figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal')
ax.imshow(vMagGrid, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=plt.get_cmap('jet') , interpolation='none')
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="2%", pad=0.2)
fig.colorbar(im, cax = cax1)
im.set_clim(0,velocityMaxNorm)
ax.set_aspect('equal')
ax.set_title(r'$||\mathbf{v}||$')
ax.set_xlabel("$x$ $[m]$")
ax.set_ylabel("$y$ $[m]$")
fig.set_tight_layout("true")

weights = np.ones(vx.shape)/vx.size
figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
#fig.subplots_adjust(left=0.075, right=0.975, bottom=0.05, top=0.95, wspace=0.2, hspace=0.25)
ax = fig.add_subplot(111)
ax.hist(vMag,100,weights=weights,histtype='step')
ax.hist(vx,100,weights=weights,histtype='step')
ax.hist(vy,100,weights=weights,histtype='step')
ax.set_xlabel('velocity')
ax.set_ylabel('tie point fraction')
ax.set_title('velocity histograms')
ax.legend([r'$||\mathbf{v}||$','$v_x$','$v_y$'])
fig.set_tight_layout("true")

figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111)
ax.hist(np.sqrt(pixelVx**2+pixelVy**2),100,weights=weights,histtype='step')
ax.hist(pixelVx,100,weights=weights,histtype='step')
ax.hist(pixelVy,100,weights=weights,histtype='step')
ax.set_xlabel('velocity*maxDeltaT (pixels)')
ax.set_ylabel('tie point fraction')
ax.set_title('pixel offset histograms (search range)')
ax.legend([r'$||\mathbf{v}||$','$v_x$','$v_y$'])
fig.set_tight_layout("true")

#figCount+=1
#fig = plt.figure(figCount, figsize=[width,height])
#ax = fig.add_subplot(111, aspect='equal')
#plt.plot(x[maskXAxis], vy[maskXAxis], '.k',gx,gridVy[xAxisGridIndex,:],'r')
#plt.set_title('$v_y$ along $x$ axis within $dy = %.1f$ of $y = %.1f$'%(dy,y0))
#plt.set_xlabel('$x$ $[m]$')
#plt.set_ylabel('$v_y$ $[m/s]$')
#plt.axis('tight')

figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111)
ax.plot(vMag[maskYAxis], y[maskYAxis], '.k',vMagGrid[:,yAxisGridIndex],gy,'b')
ax.set_title('$||\mathbf{v}||$ along $y$ axis within $dx = %.1f$ of $x = %.1f$'%(dx,x0))
ax.set_xlabel('$||\mathbf{v}||$ $[m/s]$')
ax.set_ylabel('$y$ $[m]$')
fig.set_tight_layout("true")



figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111)
ax.plot(x[maskXAxis], vMag[maskXAxis], '.k',gx,vMagGrid[xAxisGridIndex,:],'r')
ax.set_title('$||\mathbf{v}||$ along $x$ axis within $dy = %.1f$ of $y = %.1f$'%(dy,y0))
ax.set_ylabel('$||\mathbf{v}||$ $[m/s]$')
ax.set_xlabel('$y$ $[m]$')
fig.set_tight_layout("true")


#plot masked velocity with velocityMask
figCount+=1
fig = plt.figure(figCount, figsize=[width,height])
ax = fig.add_subplot(111, aspect='equal', axisbg=axisBG)
absVelMasked = vMagGrid * imageVelocityMask;
im = ax.imshow(absVelMasked, origin='lower', extent=(bounds[0],bounds[1],bounds[2],bounds[3]), cmap=plt.get_cmap('jet') , interpolation='none')
im.set_clim(0,velocityMaxNorm)
divider = make_axes_locatable(ax)
cax1 = divider.append_axes("right", size="2%", pad=0.2)
fig.colorbar(im, cax = cax1)
ax.set_aspect('equal')
ax.set_title(r'Velocity $||\mathbf{v}||$ , grid,  masked')
ax.set_xlabel('$x$ $[m]$')
ax.set_ylabel('$y$ $[m]$')
cax1.set_title(r'$[m/s]$')
fig.set_tight_layout("true")



if residualsFound:
  maxVal = 6.0*np.median(correlationVelocityResiduals)
  figCount+=1
  fig = plt.figure(figCount, figsize=[width,height])
  ax = fig.add_subplot(111)
  weights = np.ones(correlationVelocityResiduals.shape)/correlationVelocityResiduals.size
  ax.hist(correlationVelocityResiduals,100,range=[0.0,maxVal], weights=weights,histtype='step')
  ax.set_xlabel('correlation velocity uncertainty')
  ax.set_ylabel('tie point fraction')
  
  maxVal = 6.0*np.median(correlationLocationResiduals)
  figCount+=1
  fig = plt.figure(figCount, figsize=[width,height])
  ax = fig.add_subplot(111)
  weights = np.ones(correlationLocationResiduals.shape)/correlationLocationResiduals.size
  ax.hist(correlationLocationResiduals,100,range=[0.0,maxVal], weights=weights,histtype='step')
  ax.set_xlabel('correlation location uncertainty')
  ax.set_ylabel('tie point fraction')





plt.draw()
if options.savePlots:
  for index in range(1,figCount+1):
    outFileName = '%s/%s%03i%s' %(folder,options.figurePrefix, index,options.figureExt)
    plt.figure(index)
    plt.savefig(outFileName)
else:
  plt.show()

