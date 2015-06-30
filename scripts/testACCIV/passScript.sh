#!/bin/bash
accivHome='/home/zfmgpu/Desktop/Repository/acciv/'
acciv=$accivHome/acciv/acciv.exe

$accivHome/acciv/makeGeometryFactors.exe image000.h5 gridGeometryFactors.h5 flat
passes=(test/pass1 test/pass2 test/pass3)
res=0
for pass in  "${passes[@]}"; do
    
    if [ $res -eq 0 ]; then
        echo "Do pass $pass"
        $acciv $pass
        res=$?
        echo "Do pass $pass plot"
        python testScripts/plotVelocities.py --folder=$pass --imageFileName=image000.h5 --savePlots
    else
        echo "Exit-> passes failed!"
        exit 1
    fi
    
done


#lastPass=${passes[${#passes[@]} - 1]}
#python testScripts/plotVelocities.py --folder=$pass --imageFileName=image000.h5 
#python testScripts/plotAdvectedImages.py --inFolder=$pass/_work --outFolder=$pass