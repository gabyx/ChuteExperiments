#!/bin/bash
accivHome='/home/zfmgpu/Desktop/Repository/acciv/'
acciv=$accivHome/acciv/acciv.exe

$accivHome/acciv/makeGeometryFactors.exe image000.h5 gridGeometryFactors.h5 flat
passes=(test/pass1 test/pass2 test/pass3)
for pass in  "${passes[@]}"; do
    echo "Do pass $pass"
    $acciv $pass
    
    python testScripts/plotVelocities.py --folder=$pass --imageFileName=image000.h5 --savePlots
done

#lastPass=${passes[${#passes[@]} - 1]}
#python testScripts/plotVelocities.py --folder=$pass --imageFileName=image000.h5 
#python testScripts/plotAdvectedImages.py --inFolder=$pass/_work --outFolder=$pass