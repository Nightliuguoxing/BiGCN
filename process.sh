#! /bin/bash
rz=`date +%Y%m%d`
nohup python ./Process/getWeibograph.py > ./logs/P${rz}.log 2>&1 &