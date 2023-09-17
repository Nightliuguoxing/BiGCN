#! /bin/bash
unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
python ./Process/getWeibograph.py
python ./model/Weibo/BiGCN_Weibo.py 100
