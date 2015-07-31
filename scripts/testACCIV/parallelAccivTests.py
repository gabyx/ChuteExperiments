#!/usr/bin/env python 

import os, glob, re,subprocess,traceback,logging,time,multiprocessing
from multiprocessing import Pool, Manager
from progressbar import Bar, Percentage, ProgressBar
from attrdict import AttrMap

from importlib.machinery import SourceFileLoader
l = SourceFileLoader("MultiProcessingLog","../common/MultiProcessingLog.py")
mpl = l.load_module()

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-t", "--testsFolder", dest="testsFolder", help="Tests folder", metavar="FILE")

args = parser.parse_args()
settings = AttrMap(vars(args));


if not settings.testsFolder or not settings.testsFolder:
    raise NameError("Specify the folder where all tests are!");


def subDirs(d):
    return list(filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)]))

tests = subDirs(settings.testsFolder)
print("Performing tests in parallel:" + str(tests) )



def executeTest(f):
    logging.info("Process %i executing test: %s " % (os.getpid(),f)  )
    subprocess.call(['./accivTest.sh', f])
    
    


mplog = mpl.MultiProcessingLog("parallelAccivTests.log", mode='w+')
mplog.setFormatter( logging.Formatter('%(asctime)s - %(lineno)d - %(levelname)-8s - %(message)s') )
logger = logging.getLogger()
logger.addHandler(mplog)
logger.setLevel(logging.DEBUG)

pool = Pool();
mgr = Manager();
queue = mgr.Queue();

# map converter.process function over all tests
result = pool.map_async(executeTest, tests);

# monitor loop
pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(tests) ).start();
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
    