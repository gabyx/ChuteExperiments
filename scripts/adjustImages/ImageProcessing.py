#!/usr/bin/env python 


import os, glob, re,subprocess,traceback,logging
import time

import multiprocessing
from multiprocessing import Pool, Manager

from progressbar import Bar, Percentage, ProgressBar
from attrdict import AttrMap

import numpy as np
import h5py

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import scipy as scp
import skimage.exposure
from skimage.io import imread
from skimage.filters import *
from skimage.morphology import *
from skimage import data
from skimage import img_as_float
from skimage.morphology import disk
from skimage import measure

from MultiProcessingLog import MultiProcessingLog


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-i", "--inputFolder", dest="inputFolder", help="Input folder", metavar="FILE")
parser.add_argument("-o", "--outputFolder", dest="outputFolder", help="Output folder  (.tiff or something similar)", metavar="OUTPUT")
parser.add_argument("-f", "--inputRegex", dest="inputRegex", help="Regex to filter filesnames for input", metavar="JPG/TIF")
parser.add_argument("-b", "--background", dest="background", help="Background image to subtract", metavar="JPG/TIF", required=True )
parser.add_argument("--startIdx", type=int, dest="startIdx", help="Start frame number, specified as P<frameIdx> in inputRegex", metavar="JPG/TIF", default=0)
parser.add_argument("--endIdx",  type=int, dest="endIdx", help="Start frame number, specified as (?P<frameIdx>) in inputRegex", metavar="JPG/TIF", default=-1)

parser.add_argument("-r", "--outputReplacement", dest="outputReplacement", help="Replacement string to generate new file name, can contain backreferences!", metavar="JPG/TIF")                   
                  

args = parser.parse_args()
settings = AttrMap(vars(args));

if not settings.inputFolder or not settings.outputFolder:
    raise NameError("Specify input and output folder!");

if not settings.inputRegex :
    settings.inputRegex = ".*/(.*)-(?P<frameNr>\d*)\.jpg";
print( "settings.inputRegex: ", settings.inputRegex)
if not settings.outputReplacement :
    settings.outputReplacement = r"\1-%s-\2.jpg";
print( "settings.outputReplacement: ", settings.outputReplacement)
    
settings.inputRegex = re.compile(settings.inputRegex )

files = glob.glob(settings.inputFolder + '/*')
if settings.endIdx == -1:
    settings.endIdx = 1000000000
    
print( "Start Idx: %i " % settings.startIdx )
print( "End Idx: %i " % settings.endIdx )

def excludeRange(f):
    m = settings.inputRegex.match(f);
    if m:
        if int(m.group("frameNr")) in range(settings.startIdx,settings.endIdx):
            return True;
    
    return False;
    
files = list(filter(excludeRange, files));

# =================================================

settings = AttrMap(vars(args));

settings.pxPerMeter = 967.5931
settings.dt=0.001;
settings.background = imread( settings.background , as_grey = True)


# =================================================




class ConvertImage:
    
    def __init__(self,settings, queue, logger=None):
        self.logger = logger
        self.sett = settings
        self.queue = queue
    
    def __call__(self,f):
        self.process(f)
    
    def process(self,f):
        try:
            
            background = settings.background
            
            logging.info("Process %i" % os.getpid() )
            
            out = os.path.join( self.sett.outputFolder , self.sett.inputRegex.sub(self.sett.outputReplacement, f) )
            logging.info("Process image: " + f + " ---> " + out );
        
            # =========================================================================
            image =  imread(f, as_grey = True)
            image_clahe = skimage.exposure.equalize_adapthist(image,ntiles_x=16,ntiles_y=16);
            

            # background subtract and make binary mask
            # image_clahe = skimage.exposure.equalize_adapthist(image,ntiles_x=16,ntiles_y=16);

            # background subtract and make binary mask
            subtracted = np.abs(image - background)
            th = threshold_otsu(subtracted)
            binary = (subtracted > th).astype(np.uint8)

            d = disk(5)
            binary2 = binary_opening(binary, d)
            binary3 = subtracted > 0.5*th

            denoise = remove_small_objects(binary3,min_size=5000) # inplace = True
            dilated = binary_dilation(denoise, rectangle(4,8))
            blurred = gaussian_filter(dilated,sigma=1) # inplace = True

            masked = denoise * image

            # find max. length contour (interpolate)
            contours = measure.find_contours(blurred, 0.5)
            if contours :
                 contourMax = max(contours,key=lambda x: x.shape[0])
                 tck,u = scp.interpolate.splprep([contourMax[:, 0], contourMax[:, 1]], s=1e4)
                 unew = np.linspace(0, 1.0, 3000)
                 contourInterp = scp.interpolate.splev(unew, tck)
            else:
                 contourMax = None
                 contourInterp = None

            
            # ====================================================================================================
            
            dpi=100
            # Make Contour Frames
            f = plt.figure(num=None, frameon=False, figsize=(self.sett.background.shape[1]/dpi,
                                                             self.sett.background.shape[0]/dpi), dpi=dpi)
            ax = plt.Axes(f, [0., 0., 1., 1.])
            #ax.set_axis_off()
            f.add_axes(ax)
            ax.imshow(image,cmap=cm.Greys_r)
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
            if contourInterp:
               ax.plot(contourInterp[1], contourInterp[0], "b", linewidth=2)
            f.savefig(out % "contour")
            plt.close(f)
            f=None
            ax=None
            
            # Make Mask Frames
            f = plt.figure(num=None, frameon=False, figsize=(self.sett.background.shape[1]/dpi,
                                                             self.sett.background.shape[0]/dpi), dpi=dpi)
            ax = plt.Axes(f, [0., 0., 1., 1.])
            #ax.set_axis_off()
            f.add_axes(ax)
            ax.imshow(dilated,cmap=cm.Greys_r)
            ax.set_autoscalex_on(False)
            ax.set_autoscaley_on(False)
            f.savefig(out % "mask")
            plt.close(f)
            f=None
            ax=None
                
             
            #     time = count*dt;
            #     h5File = h5py.File('image%03i.h5'%count, 'w')
            #     dataset = h5File.create_dataset("bounds", data=np.array(bounds))
            #     dataset = h5File.create_dataset("data", data=image_clahe)
            #     dataset = h5File.create_dataset("mask", data=mask)
            #     dataset = h5File.create_dataset("time", data=np.array(time))
            #     h5File.close()
                
            self.queue.put(f);
        except Exception as e:
            logging.info("Exception: " + e.message)
            exc_buffer = io.StringIO()
            traceback.print_exc(file=exc_buffer)
            logging.info(exc_buffer.getvalue())
            raise e
        except:
            logging.info("Exception!")
            exc_buffer = io.StringIO()
            traceback.print_exc(file=exc_buffer)
            logging.info(exc_buffer.getvalue())
            raise
            

mpl = MultiProcessingLog("ImageProcessing.log", mode='w+')
mpl.setFormatter( logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)-8s - %(message)s') )
logger = logging.getLogger()
logger.addHandler(mpl)
logger.setLevel(logging.DEBUG)


pool = Pool();
mgr = Manager();
queue = mgr.Queue();

converter = ConvertImage(settings, queue)

# map converter.process function over all files
result = pool.map_async(converter, files);

# monitor loop
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(files) ).start();
while True:
    if result.ready():
        break
    else:
        pbar.update(queue.qsize());
        time.sleep(0.1)
    
pbar.finish()

pool.close();
pool.join()

logging.shutdown()