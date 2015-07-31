#!/bin/bash

accivBUILD='/home/zfmgpu/Desktop/Repository/BUILD/acciv/acciv'
acciv=$accivBUILD/acciv

# that is done once!
#$accivBUILD/acciv-makeGeometryFactors testData/image000.h5 testData/gridGeometryFactors.h5 flat

# first argument is test folder

testFolder=$1

LOG="$testFolder/acciv.log"
echo "  Acciv Process Log: " > $LOG

if [ -z "$testFolder" ]; then
    echo "No test folder given: $1" >> $LOG
    exit 1
fi

echo "TEST $testFolder ==============================================================" >> $LOG
res=0
for pass in  $testFolder/*/; do
    
    if [ $res -eq 0 ]; then
        echo "Do pass $pass" >> $LOG
        $acciv $pass #>> $LOG 2>&1
        res=$?
    else
        echo "Exit-> passes failed!" >> $LOG
        exit 1
    fi
    
    if [ $res -eq 0 ]; then 
        echo "Do pass $pass plot" >> $LOG
        python testScripts/plotVelocities.py --folder=$pass --imageFileName=testData/image000.h5 --savePlots >> $LOG 2>&1
        python testScripts/plotResiduals.py  --folder=$pass --scatterFileName=testData/outScatteredVelocity.h5 --savePlots >> $LOG 2>&1
    else
        echo "Exit-> passes failed!" >> $LOG
        exit 1
    fi
    
done
