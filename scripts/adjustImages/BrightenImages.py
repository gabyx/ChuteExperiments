#!/usr/bin/env python 


import os, glob, re,subprocess;
import time
from multiprocessing import Pool, Manager
from progressbar import Bar, Percentage, ProgressBar


from wand.image import Image


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", "--inputFolder", dest="inputFolder", help="Input folder", metavar="FILE")
parser.add_option("-f", "--inputRegex", dest="inputRegex", help="Regex to filter filesnames for input", metavar="JPG/TIF")
parser.add_option("-r", "--outputReplacement", dest="outputReplacement", help="Replacement string to generate new file name, can contain backreferences!", metavar="JPG/TIF")                   
parser.add_option("-c", "--command", dest="command", help="ImageMagick command options!", metavar="JPG/TIF")                   

(opts, args) = parser.parse_args()

if not opts.inputFolder or not opts.outputFolder:
    raise NameError("Specify input and output folder!");

if not opts.inputRegex :
    opts.inputRegex = "(.*?)(\d*)\.jpg";
print( "opts.inputRegex: ", opts.inputRegex)
if not opts.outputReplacement :
    opts.outputReplacement = r"\1gray_\2.jpg";
print( "opts.outputReplacement: ", opts.outputReplacement)
    
inputFileRegex = re.compile(opts.inputRegex )

files = glob.glob(opts.inputFolder + '*.jpg')
files = list(filter(inputFileRegex.match, files));



class ConvertImage:
    
    def __init__(self,command, inputFileRegex, outputReplacement, queue):
        self.command=command;
        self.inputFileRegex = inputFileRegex;
        self.outputReplacement = outputReplacement;
        self.queue = queue;
    
    def process(self,f):
        out = self.inputFileRegex.sub(self.outputReplacement, f);
        
        #print("Convert image: " + f + " ---> " + out );
        command = ["convert"]
        command += [str(f)];
        command += self.command.split(" ");
        command += [str(out)];
        #print(command)
        subprocess.call(command);
        
        # do image manipulation here
        #  with Image(filename=f) as img:
        # img = Image(filename=f);
        #img.colorspace="gray";
        #img.adjust_levels()
        #img.save(filename=out);
        
        # put filename into finished file queue
        self.queue.put(f);


pool = Pool();
mgr = Manager();
queue = mgr.Queue();
converter = ConvertImage(opts.command,inputFileRegex,opts.outputReplacement, queue)

# map converter.process function over all files
result = pool.map_async(converter.process, files);


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